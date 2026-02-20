import os
import json
import logging
import uuid
import pandas as pd
import plotly.express as px
from openai import OpenAI
from ingest_excel import get_embedding, get_db_connection
import rbac

# NOTE: Agent imports are done lazily inside functions to avoid circular imports
# retrieval.py <- agents/__init__.py <- query_agent.py <- retrieval.py (circular!)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_relevant_sheets(query, db_url, openai_key, user_role, limit=3):
    """
    Find relevant sheets using vector similarity + RBAC filtering.
    """
    embedding = get_embedding(query, openai_key)
    if not embedding:
        return []

    conn = get_db_connection(db_url)
    try:
        cur = conn.cursor()
        # Searching summary_embedding using cosine distance
        # Hybrid Search: Vector Distance + Filename Keyword match
        # If the query string overlaps significantly with the filename, boost it.
        # Simple heuristic: If file_name contains the full query (rare) or query contains file_name.
        
        sql = """
            SELECT 
                s.sheet_id, 
                s.table_name, 
                s.sheet_name, 
                s.category, 
                s.columns_metadata,
                s.data_domain,
                s.sensitivity_level,
                f.file_name,
                COALESCE(p.allow_aggregation, true)  AS allow_aggregation,
                COALESCE(p.allow_raw_rows, false)     AS allow_raw_rows,
                (s.summary_embedding <=> %s::vector) as distance
            FROM sheets_metadata s
            JOIN files_metadata f ON s.file_id = f.file_id
            LEFT JOIN retrieval_policies p 
              ON p.data_domain = s.data_domain
             AND p.sensitivity_level = s.sensitivity_level
             AND p.role = %s
             AND p.allowed = true
            WHERE (p.policy_id IS NOT NULL OR s.data_domain IS NULL OR s.sensitivity_level IS NULL)
            ORDER BY 
                (CASE WHEN f.file_name ILIKE %s THEN 0 ELSE 1 END) ASC,
                distance ASC
            LIMIT %s
        """
        # Prepare fuzzy match param
        like_query = f"%{query}%"
        
        cur.execute(sql, (embedding, user_role, like_query, limit))
        results = []
        for row in cur.fetchall():
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
        return results
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return []
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
    
    Core Rules:
    1. Use ONLY the table and columns provided in the schema.
    2. Use ILIKE for flexible text matching (e.g. WHERE vendor ILIKE '%amazon%').
    3. Return ONLY the raw SQL query. No markdown, no explanations.
    
     ARITHMETIC OPERATIONS — Follow these patterns:
    
    | User asks...                           | SQL to generate                                               |
    |----------------------------------------|---------------------------------------------------------------|
    | total / sum of a column                | SUM(column)                                                   |
    | average / mean                         | AVG(column)                                                   |
    | difference / minus / subtract          | SUM(revenue_col) - SUM(expense_col) AS net                    |
    | profit margin / percentage             | ROUND(SUM(profit_col) * 100.0 / NULLIF(SUM(revenue_col),0),2) AS margin_pct |
    | ratio between two columns              | ROUND(SUM(col_a)::numeric / NULLIF(SUM(col_b),0), 4) AS ratio |
    | growth / change compared to last year  | Use LAG() or subquery to compute period-over-period change    |
    | multiply columns                       | SUM(quantity_col * price_col) AS total_value                  |
    | divide / per unit                      | ROUND(SUM(amount_col)::numeric / NULLIF(SUM(units_col),0), 2) AS per_unit |
    | cumulative / running total             | SUM(col) OVER (ORDER BY date_col) AS running_total            |
    | rank / top N                           | ORDER BY metric_col DESC LIMIT N                              |
    | standard deviation / variance          | STDDEV(column), VARIANCE(column)                              |
    | min and max range                      | MAX(col) - MIN(col) AS range                                  |
    
    ARITHMETIC RULES:
    - Always use NULLIF(divisor, 0) to avoid division-by-zero errors.
    - Always cast to ::numeric before division to avoid integer truncation.
    - Use ROUND(..., 2) for percentages and ratios.
    - Use aliases (AS) to name calculated columns clearly.
    - For multi-step calculations, use a CTE (WITH clause) for clarity.
    
    EXAMPLE — "What is the profit margin by category?":
    SELECT category,
           SUM(revenue) AS total_revenue,
           SUM(cost) AS total_cost,
           SUM(revenue) - SUM(cost) AS profit,
           ROUND((SUM(revenue) - SUM(cost)) * 100.0 / NULLIF(SUM(revenue), 0), 2) AS margin_pct
    FROM {table_name}
    GROUP BY category
    ORDER BY margin_pct DESC;
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
        return "The query returned no results."
    
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
    
    # 1. Search (Fetch top 3 instead of 1 to handle cases where top match is wrong)
    sheets = search_relevant_sheets(user_query, db_url, openai_key, user_role, limit=3)
    if not sheets:
        return {"error": "No relevant data found for your query. (Check if file is uploaded and you have access)"}
    
    candidate_results = []
    
    # Iterate through potential matches
    for i, sheet in enumerate(sheets):
        logging.info(f"Trying sheet match #{i+1}: {sheet['file_name']} - {sheet['sheet_name']}")
        
        # Check if this is a metadata question ("Which file...")
        if is_metadata_query(user_query):
            # ... (Metadata logic remains same)
            steps['sheet_match'] = sheet
            steps['generated_sql'] = "-- Metadata lookup"
            df = pd.DataFrame([{
                "File Name": sheet['file_name'],
                "Sheet Name": sheet['sheet_name'],
                "Table Name": sheet['table_name'],
                "Category": sheet['category'],
                "Confidence": f"{1 - sheet['distance']:.2f}"
            }])
            steps['results_df'] = df
            steps['final_answer'] = f"Found relevant data in **{sheet['file_name']}** (Sheet: {sheet['sheet_name']})."
            return steps

        # 2. Generate SQL for this sheet
        try:
            sql = generate_sql_query(user_query, sheet, openai_key)
            
            # 2b. Enforce RBAC
            from agents.rbac_agent import RBACAgent
            from agents.base_agent import AgentTask
            rbac_agent = RBACAgent(db_url=db_url)
            rbac_task = AgentTask(
                task_id=str(uuid.uuid4()),
                task_type="validate_sql",
                payload={"sql_query": sql, "sheet_info": sheet}
            )
            rbac_result = rbac_agent.execute(rbac_task)
            if rbac_result.status.value == "failed":
                logging.warning(f"RBAC failed for {sheet['file_name']}: {rbac_result.error}")
                continue
                
        except Exception as e:
            logging.warning(f"SQL Gen failed for {sheet['file_name']}: {e}")
            continue

        # 3. Execute
        df, error = execute_query(sql, db_url)
        
        if error:
            logging.warning(f"SQL execution error on {sheet['file_name']}: {error}")
            continue
            
        if df is not None:
            # Check for "meaningful" data (not just 0 or null)
            is_meaningful = False
            if not df.empty:
                # Check if any value is non-zero/non-null
                # For single row/col results (aggregations):
                if df.shape == (1, 1):
                    val = df.iloc[0, 0]
                    if pd.notna(val) and val != 0 and val != "0":
                        is_meaningful = True
                else:
                    is_meaningful = True # Multiple rows usually mean data found
            
            candidate_results.append({
                "sheet": sheet,
                "sql": sql,
                "df": df,
                "meaningful": is_meaningful
            })
            
            # Optimization: If we found meaningful data in the *very first* (best) match, 
            # we can trust it and stop early.
            if i == 0 and is_meaningful:
                break

    # Decision Logic: Select Best Candidate
    if not candidate_results:
        steps['error'] = "Could not find valid results in any relevant file."
        return steps
        
    # Prioritize meaningful (non-zero) results
    best_candidate = None
    
    # 1. Look for meaningful result
    for cand in candidate_results:
        if cand['meaningful']:
            best_candidate = cand
            break
            
    # 2. If no meaningful result, fallback to the first successful execution (even if 0)
    if not best_candidate:
        best_candidate = candidate_results[0]
        
    # Finalize steps with best candidate
    steps['sheet_match'] = best_candidate['sheet']
    steps['generated_sql'] = best_candidate['sql']
    steps['results_df'] = best_candidate['df']
    
    # 4. Synthesize
    answer = synthesize_answer(user_query, best_candidate['sql'], best_candidate['df'], best_candidate['sheet'], openai_key)
    steps['final_answer'] = answer
    
    # 5. Charting using ChartAgent (lazy import to avoid circular)
    if df is not None and not df.empty:
        from agents.chart_agent import ChartAgent
        from agents.base_agent import AgentTask
        chart_agent = ChartAgent(openai_key=openai_key)
        chart_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="generate_visualization",
            payload={
                "user_query": user_query,
                "dataframe": df
            }
        )
        chart_response = chart_agent.execute(chart_task)
        
        if chart_response.status.value == "completed" and chart_response.result:
            steps['chart_spec'] = chart_response.result.get('chart_spec')
            if chart_response.result.get('chart'):
                steps['chart'] = chart_response.result['chart']

    return steps
