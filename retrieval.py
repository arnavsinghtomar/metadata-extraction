import os
import json
import logging
import pandas as pd
import plotly.express as px
from openai import OpenAI
from ingest_excel import get_embedding, get_db_connection
import rbac

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_relevant_sheets(query, db_url, openai_key, user_role, limit=3):
    """
    Find relevant sheets using vector similarity + RBAC filtering.
    Returns (results, diagnostics)
    """
    logging.info(f"🔍 Searching relevant sheets for Query: '{query}' | User Role: {user_role}")
    embedding = get_embedding(query, openai_key)
    if not embedding:
        logging.error("❌ Failed to generate embedding for the query.")
        return []

    conn = get_db_connection(db_url)
    try:
        cur = conn.cursor()
        # Searching summary_embedding using cosine distance
        # Hybrid Search: Vector Distance + Filename Keyword match
        # If the query string overlaps significantly with the filename, boost it.
        # Simple heuristic: If file_name contains the full query (rare) or query contains file_name.
        
        sql = """
            WITH sensitivity_ranks AS (
                SELECT 'public' as level, 1 as rank
                UNION ALL SELECT 'internal', 2
                UNION ALL SELECT 'confidential', 3
                UNION ALL SELECT 'restricted', 4
            )
            SELECT DISTINCT ON (s.sheet_id)
                s.sheet_id, 
                s.table_name, 
                s.sheet_name, 
                s.category, 
                s.columns_metadata,
                s.data_domain,
                s.sensitivity_level,
                f.file_name,
                p.allow_aggregation,
                p.allow_raw_rows,
                (s.summary_embedding <=> %s::vector) as distance
            FROM sheets_metadata s
            JOIN files_metadata f ON s.file_id = f.file_id
            JOIN sensitivity_ranks sr_s ON s.sensitivity_level = sr_s.level
            JOIN retrieval_policies p ON p.data_domain = s.data_domain
            JOIN sensitivity_ranks sr_p ON p.sensitivity_level = sr_p.level
            WHERE p.role = %s
              AND p.allowed = true
              AND sr_p.rank >= sr_s.rank
            ORDER BY 
                s.sheet_id,
                sr_p.rank ASC, -- Pick the least restrictive applicable policy
                (CASE WHEN f.file_name ILIKE %s THEN 0 ELSE 1 END) ASC,
                distance ASC
        """
        # Prepare fuzzy match param
        like_query = f"%{query}%"
        
        # We need to wrap the final results to re-sort by distance since DISTINCT ON required sorting by sheet_id first
        wrapped_sql = f"""
            SELECT * FROM ({sql}) sub
            ORDER BY (CASE WHEN file_name ILIKE %s THEN 0 ELSE 1 END) ASC, distance ASC
            LIMIT %s
        """
        
        cur.execute(wrapped_sql, (embedding, user_role, like_query, like_query, limit))
        results = []
        rows = cur.fetchall()
        logging.info(f"📊 Query executed. Found {len(rows)} candidate sheets matching Role & Keyword filters.")
        
        for row in rows:
            results.append({
                "sheet_id": row[0],
                "table_name": row[1],
                "sheet_name": row[2],
                "category": row[3],
                "columns_metadata": row[4], # JSONB
                "data_domain": row[5],
                "sensitivity_level": row[6],
                "file_name": row[7],
                "allow_aggregation": row[8],
                "allow_raw_rows": row[9],
                "distance": row[10]
            })
        
        if not results:
            logging.warning(f"⚠️ No sheets passed the RBAC and similarity filters for role: {user_role}")
            
            # Simple Diagnostic: Check if any sheets exist at all in the DB
            cur.execute("SELECT count(*) FROM sheets_metadata")
            total_sheets = cur.fetchone()[0]
            
            # Check if sheets exist but were blocked by RBAC for this role
            cur.execute("SELECT count(*) FROM sheets_metadata s JOIN files_metadata f ON s.file_id = f.file_id")
            # Without RBAC filter
            # (In a real scenario we might check for 'close' embedding matches that were filter out)
            
            diag = {
                "total_sheets_in_db": total_sheets,
                "role": user_role,
                "reason": "No matches found after RBAC filtering." if total_sheets > 0 else "Database is empty."
            }
            return [], diag
            
        return results, {}
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return [], {"error": str(e)}
    finally:
        conn.close()

def generate_sql_query(user_query, sheet_info, openai_key):
    """
    Generate SQL for a specific sheet given its schema.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openai_key
    )
    
    table_name = sheet_info['table_name']
    columns_meta = sheet_info.get('columns_metadata', [])
    
    # Build schema description
    schema_desc = f"Table: {table_name}\nColumns:\n"
    
    # Handle if it's already a dict/list or string
    if isinstance(columns_meta, str):
        try:
            columns_meta = json.loads(columns_meta)
        except:
            columns_meta = []
            
    if columns_meta:
        for col in columns_meta:
            c_name = col.get('name')
            c_orig = col.get('original_name')
            c_type = col.get('type')
            c_samples = col.get('samples', [])
            
            schema_desc += f"- {c_name} ({c_type})\n"
            schema_desc += f"  Original Header: {c_orig}\n"
            if c_samples:
                schema_desc += f"  Sample Values: {', '.join(str(s) for s in c_samples)}\n"
    else:
        schema_desc += "(No detailed column metadata available. Use generic queries.)"
        
    # Get access control constraints
    allow_raw = sheet_info.get('allow_raw_rows', False)
    allow_agg = sheet_info.get('allow_aggregation', True)
    
    # Build access control rules
    access_rules = "\n🔐 CRITICAL ACCESS CONTROL RULES:\n"
    if not allow_raw:
        access_rules += """
- You are NOT allowed to return raw row-level data
- You MUST use aggregation functions: SUM, AVG, COUNT, MAX, MIN, GROUP BY
- NEVER use SELECT * or individual row selection
- Example: Instead of "SELECT vendor, amount FROM table", use "SELECT vendor, SUM(amount) FROM table GROUP BY vendor"
"""
    
    if not allow_agg and not allow_raw:
        access_rules += "- User has NO access to this data domain. Return error.\n"

    prompt = f"""
    You are a PostgreSQL expert assisting a user with a financial query.
    
    Target Data Dictionary:
    {schema_desc}
    
    User Question: "{user_query}"
    
    {access_rules}
    
    Goal: Write a valid PostgreSQL query to answer the question while respecting access rules.
    
    Rules:
    1. Use ONLY the table and columns provided in the schema.
    2. Use ILIKE for flexible text matching (e.g. WHERE vendor ILIKE '%amazon%').
    3. If the user asks for a total, use SUM/COUNT.
    4. Return ONLY the raw SQL query. No markdown (```sql), no explanations.
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a SQL expert. Output raw SQL only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    
    sql_query = response.choices[0].message.content.strip()
    
    # Clean up formatting if GPT still adds it
    if sql_query.startswith("```"):
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
    return sql_query

def execute_query(sql, db_url):
    """
    Execute the SQL query and return a pandas DataFrame.
    """
    if not sql:
        return None, "No SQL generated"
        
    conn = get_db_connection(db_url)
    try:
        # Use pandas read_sql to easily get columns and data
        df = pd.read_sql(sql, conn)
        return df, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

def is_metadata_query(query):
    """Check if the user is asking about file/sheet existence rather than data."""
    q = query.lower().strip()
    triggers = ["which file", "what file", "which sheet", "what sheet", "where is", "source of"]
    return any(q.startswith(t) for t in triggers)


def synthesize_answer(user_query, sql, df, sheet_info, openai_key):
    """
    Generate a natural language answer based on the query results with citations.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openai_key
    )
    
    if df is None or df.empty:
        prompt = f"""
        The user asked: "{user_query}"
        
        We tried to find the answer by running this SQL:
        {sql}
        
        However, the query returned NO results. 
        Based on the user's question and the SQL, please explain in a friendly, natural way why there might be no data.
        Possible reasons to consider:
        - The specific filters (WHERE clause) might be too restrictive.
        - The requested time period or category might not be in the data.
        - The user might be asking for something that isn't tracked in this specific table.
        
        Do not say "I don't know". Instead, suggest how the user could broaden their search or what might be missing.
        """
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst explaining why a search returned no results."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    
    # Create a compact string representation of the data
    if len(df) > 10:
        data_summary = df.head(10).to_markdown(index=False)
        data_summary += f"\n... ({len(df)-10} more rows)"
    else:
        data_summary = df.to_markdown(index=False)
        
    prompt = f"""
    The user asked: "{user_query}"
    
    We executed this SQL: 
    {sql}
    
    And got these results:
    {data_summary}
    
    Please provide a concise, natural language answer to the user's question based on these results. 
    If the data is a table of rows, summarize the key findings or mention what is listed. 
    If it's a single number, state it clearly.
    Respond in a structured, user-friendly format with headings and bullet points.

    CRITICAL INSTRUCTIONS:
    1. If the user asks "Which file...", start your answer by explicitly stating the file name.
    2. Example: "The data containing [Topic] is located in the file **[File Name]** (Sheet: [Sheet Name])."
    3. Then provide the answer derived from the SQL results.
    
    At the end of your response, add a section titled "### 📚 Data sources" and include:
    - **Source File:** {sheet_info.get('file_name', 'Unknown')}
    - **Sheet Name:** {sheet_info.get('sheet_name', 'Unknown')}
    - **DB Table:** {sheet_info.get('table_name', 'Unknown')}
    """
    
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt}
        ],
        
        temperature=0.3
    )
    
    return response.choices[0].message.content

def decide_chart(user_query, df, openai_key):
    """
    Decide if a visualization is useful and return chart specification.
    """
    print("deciding chart")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openai_key
    )

    columns_info = ", ".join(
        [f"{c} ({df[c].dtype})" for c in df.columns]
    )
    print(columns_info)

    prompt = f"""
You are an expert data analyst deciding whether to show a chart.

User Question:
"{user_query}"

Available Data Columns (ONLY these can be used):
{columns_info}

IMPORTANT DEFINITIONS:
- Categorical columns: object, string, text
- Numeric columns: int, float, double
- Time columns: any column whose name contains date, time, year, month, day
  (even if dtype is object/string)

CRITICAL RULES:
1. You MUST choose x_axis and y_axis ONLY from the listed columns.
2. DO NOT invent columns.
3. If data is wide (multiple numeric columns), choose ONE numeric column and proceed.
4. If table has ≤2 rows, show a chart if comparison is still meaningful.
5. Line charts are allowed if a time-like column exists (name-based).
6. Pie charts ONLY if categories ≤6.
7. You MUST make a best-effort decision.
   Return show_chart=false ONLY if chart is clearly impossible.

Decision logic:
- Category + numeric → bar
- Time + numeric → line
- Single numeric only → no chart
- No category or time column → no chart

Respond in STRICT JSON ONLY:

{{
  "show_chart": true,
  "chart_type": "bar",
  "x_axis": "column_name",
  "y_axis": "column_name",
  "reason": "short justification"
}}
"""


    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    print(content)

    # Extract JSON if it's wrapped in text or markdown
    try:
        if "{" in content and "}" in content:
            content = content[content.find("{"):content.rfind("}")+1]
        
        parsed_json = json.loads(content)
        print(parsed_json)
        return parsed_json
    except Exception as e:
        logging.error(f"Failed to parse chart spec: {e}")
        return {"show_chart": False}

def render_chart(df, chart_spec):
    """
    Render a Plotly chart based on the specification.
    
    """
    print("rendering chart")
    print(chart_spec)
    if not chart_spec.get("show_chart"):
        return None

    chart_type = chart_spec.get("chart_type")
    x = chart_spec.get("x_axis")
    y = chart_spec.get("y_axis")

    # Validate columns exist in DF
    if x and x not in df.columns:
        logging.error(f"Chart validation failed: {x} not in {df.columns.tolist()}")
        return None
    if y and y not in df.columns:
         logging.error(f"Chart validation failed: {y} not in {df.columns.tolist()}")
         return None

    try:
        if chart_type == "bar":
            return px.bar(df, x=x, y=y, title=chart_spec.get("reason"))
        elif chart_type == "line":
            return px.line(df, x=x, y=y, title=chart_spec.get("reason"))
        elif chart_type == "pie":
            return px.pie(df, names=x, values=y, title=chart_spec.get("reason"))
        elif chart_type == "scatter":
            return px.scatter(df, x=x, y=y, title=chart_spec.get("reason"))
    except Exception as e:
        logging.error(f"Chart rendering failed: {e}")
        return None

    return None

def process_retrieval(user_query, db_url, openai_key, user_role="Analyst"):
    """
    Orchestration function:
    1. Search relevant sheet
    2. Generate SQL
    3. Execute
    4. Synthesize
    
    Returns a dict with all steps for UI display.
    """
    steps = {}
    
    # 1. Search
    sheets, diagnostics = search_relevant_sheets(user_query, db_url, openai_key, user_role, limit=1)
    if not sheets:
        logging.warning(f"🚫 Retrieval Failed: No relevant sheets found for query: '{user_query}' (Role: {user_role})")
        
        # Ask LLM to explain why no sheets were found
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openai_key)
        explainer_prompt = f"""
        User Query: "{user_query}"
        User Role: {user_role}
        System Diagnostics: {json.dumps(diagnostics)}
        
        Please explain to the user why no matching records or files were found. 
        If it's an RBAC issue (role restriction), explain it politely. 
        If the database is empty, mention that no files have been uploaded yet.
        Be helpful and suggest what they can do next.
        """
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "system", "content": "You explain data access issues to users."},
                      {"role": "user", "content": explainer_prompt}],
            temperature=0.3
        )
        return {"error": resp.choices[0].message.content}
    
    logging.info(f"🎯 Best sheet match: {sheets[0]['sheet_name']} from file {sheets[0]['file_name']} (Distance: {sheets[0]['distance']:.4f})")
    best_sheet = sheets[0]
    steps['sheet_match'] = best_sheet
    
    # Check if this is a metadata question ("Which file...")
    if is_metadata_query(user_query):
        # Skip SQL generation for metadata questions
        steps['generated_sql'] = "-- Metadata lookup (SQL skipped)"
        # Create a fake DF containing the metadata answer
        df = pd.DataFrame([{
            "File Name": best_sheet['file_name'],
            "Sheet Name": best_sheet['sheet_name'],
            "Table Name": best_sheet['table_name'],
            "Category": best_sheet['category'],
            "Match Confidence": f"{1 - best_sheet['distance']:.2f}"
        }])
        steps['results_df'] = df
        
        # Synthesize answer directly from this metadata
        answer = f"The information you asked about is located in the file **{best_sheet['file_name']}**.\n\n" \
                 f"**Details:**\n" \
                 f"- **Sheet Name:** {best_sheet['sheet_name']}\n" \
                 f"- **Category:** {best_sheet['category']}\n" \
                 f"- **Confidence:** {1 - best_sheet['distance']:.2%}"
        steps['final_answer'] = answer
        return steps

    # 2. Generate SQL
    try:
        sql = generate_sql_query(user_query, best_sheet, openai_key)
        steps['generated_sql'] = sql
        
        # 2b. Enforce RBAC at runtime (Defense-in-depth)
        rbac.validate_sql_access(sql, best_sheet)
        
    except PermissionError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Failed to generate query: {e}"}
    
    # 3. Execute
    df, error = execute_query(sql, db_url)
    if error:
        steps['error'] = f"SQL Execution Failed: {error}"
        return steps
        
    steps['results_df'] = df
    
    # 4. Synthesize
    answer = synthesize_answer(user_query, sql, df, best_sheet, openai_key)
    steps['final_answer'] = answer
    
    # 5. Charting
    if df is not None and not df.empty:
        chart_spec = decide_chart(user_query, df, openai_key)
        steps['chart_spec'] = chart_spec
        if chart_spec.get("show_chart"):
            fig = render_chart(df, chart_spec)
            if fig:
                steps['chart'] = fig

    return steps
