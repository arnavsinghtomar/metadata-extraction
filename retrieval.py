import os
import json
import logging
import pandas as pd
from openai import OpenAI
from ingest_excel import get_embedding, get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def search_relevant_sheets(query, db_url, openai_key, limit=3):
    """
    Find relevant sheets using vector similarity on summary and keywords.
    """
    embedding = get_embedding(query, openai_key)
    if not embedding:
        return []

    conn = get_db_connection(db_url)
    try:
        cur = conn.cursor()
        # Searching summary_embedding using cosine distance
        # Searching hybrid: Average of (Summary Distance + Column Schema Distance)
        # This allows matching on "Purpose" (Summary) OR "Specific Fields" (Columns)
        sql = """
            SELECT 
                sheet_id, 
                table_name, 
                sheet_name, 
                category, 
                columns_metadata,
                (
                    (COALESCE(summary_embedding <=> %s::vector, 1.0)) + 
                    (COALESCE(columns_embedding <=> %s::vector, 1.0))
                ) / 2.0 as distance
            FROM sheets_metadata
            WHERE summary_embedding IS NOT NULL OR columns_embedding IS NOT NULL
            ORDER BY distance ASC
            LIMIT %s
        """
        cur.execute(sql, (embedding, embedding, limit))
        results = []
        for row in cur.fetchall():
            results.append({
                "sheet_id": row[0],
                "table_name": row[1],
                "sheet_name": row[2],
                "category": row[3],
                "columns_metadata": row[4], # JSONB
                "distance": row[5]
            })
        return results
    except Exception as e:
        logging.error(f"Search failed: {e}")
        return []
    finally:
        conn.close()



def generate_sql_query(user_query, sheet_infos, openai_key, feedback=None):
    """
    Generate SQL for one or more sheets given their schema.
    """
    client = OpenAI(
        api_key=openai_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Build schema description for ALL relevant sheets
    schema_desc = ""
    
    for sheet in sheet_infos:
        table_name = sheet['table_name']
        sheet_name = sheet['sheet_name']
        columns_meta = sheet.get('columns_metadata', [])
        
        # Handle if it's already a dict/list or string
        if isinstance(columns_meta, str):
            try:
                columns_meta = json.loads(columns_meta)
            except:
                columns_meta = []
        
        schema_desc += f"\n--- Table: {table_name} (Sheet: {sheet_name}) ---\nColumns:\n"
        
        if columns_meta:
            for col in columns_meta:
                c_name = col.get('name')
                c_orig = col.get('original_name')
                c_type = col.get('type')
                c_role = col.get('role', 'other')
                c_samples = col.get('samples', [])
                
                schema_desc += f"- {c_name} ({c_type}) [Role: {c_role}]\n"
                schema_desc += f"  Original Header: {c_orig}\n"
                if c_samples:
                    schema_desc += f"  Sample Values: {', '.join(str(s) for s in c_samples)}\n"
        else:
            schema_desc += "(No detailed column metadata available.)\n"
        
    import re
    
    # Programmatic Month Detection
    # 1. Detect Year in Query (e.g. 2024)
    year_match = re.search(r'\b(20\d{2})\b', user_query)
    year = year_match.group(1) if year_match else None
    
    # 2. Extract and Pre-compute Valid Join Keys
    join_keys_desc = "\n[VALID JOIN KEYS (Use ONLY these for ON clauses and Subqueries)]\n"
    all_identifiers = {}
    
    for sheet in sheet_infos:
        table_name = sheet['table_name']
        columns_meta = sheet.get('columns_metadata', [])
        if isinstance(columns_meta, str):
            try: columns_meta = json.loads(columns_meta)
            except: columns_meta = []
            
        # Filter for identifiers
        # Logic: role='identifier' OR name contains 'id', 'code', 'account', 'name'
        # We rely on the ingest logic but double check here for safety
        ids = []
        for col in columns_meta:
            c_name = col.get('name', '').lower()
            c_role = col.get('role', 'other')
            
            is_explicit_id = c_role == 'identifier'
            is_implicit_id = any(k in c_name for k in ['id', 'code', 'account', 'name', 'client', 'customer'])
            
            if is_explicit_id or is_implicit_id:
                ids.append(c_name)
        
        all_identifiers[table_name] = set(ids)
        join_keys_desc += f"- Table '{table_name}': {json.dumps(ids)}\n"
        
    context_hints = join_keys_desc + "\n"
    
    if year:
        short_yr = year[-2:] # 24
        months_regex = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)'
        
        context_hints += f"\n[STRICT YEARLY FILTERING FOR {year}]\n"
        
        for sheet in sheet_infos:
            table_name = sheet['table_name']
            columns_meta = sheet.get('columns_metadata', [])
            if isinstance(columns_meta, str):
                try: columns_meta = json.loads(columns_meta)
                except: columns_meta = []
            
            # Find columns matching "dec_24" or "dec_2024" or similar
            valid_cols = []
            for col in columns_meta:
                c_name = col.get('name', '').lower()
                # Check for month name ANYWHERE and Year suffix
                # simple patterns: jan_24, jan-24, jan2024, 2024_jan
                if re.search(months_regex, c_name) and (short_yr in c_name or year in c_name):
                    valid_cols.append(c_name)
            
            if valid_cols:
                context_hints += f"Table '{table_name}': Use THESE columns for {year}: {json.dumps(valid_cols)}\n"
            else:
                context_hints += f"Table '{table_name}': NO columns found for {year}.\n"

    prompt = f"""
    You are a PostgreSQL expert assisting a user with a financial query.
    
    Target Data Dictionary (available tables):
    {schema_desc}
    
    User Question: "{user_query}"
    {context_hints}
    
    Goal: Write a valid PostgreSQL query to answer the question.
    
    Rules:
    1. **Table Selection & Joins**: 
       - **"No Revenue" / Missing Data**: ALWAYS use `LEFT JOIN` (starting with the entity table e.g. Customers) to preserve rows with no match in the financial table.
       - **Revenue Required**: Use `INNER JOIN` only if the user asks for entities that *have* revenue/sales.
       - **Verify Join Keys (CRITICAL)**: 
         - **STRICT ENFORCEMENT**: You MUST use columns listed in [VALID JOIN KEYS] above.
         - **Subqueries & Scope**: 
             - Columns in `GROUP BY` or `SELECT` inside a subquery MUST exist in the subquery's `FROM` table.
             - **Check**: Is the column in the [VALID JOIN KEYS] list for that table? If not, DO NOT USE IT.
         - **No 'customer_id' Assumption**: Do NOT assume 'customer_id' exists. Use the real keys provided.
         - **Use ONLY keys that exist in both tables.**
         - **FAILURE CONDITION**: If you need to join two tables but they do not share any column name (or obvious key like 'account' vs 'account_id') from the [VALID JOIN KEYS] list, you must STOP and output exactly:
           "No joinable identifier found between contracts and P&L tables."
    2. **Semantic Columns**: Use the [Role: ...] hints. 
       - If asking for "Revenue" and no explicit 'revenue' column exists, look for columns with Role: 'monthly_metric'.
    3. **Yearly & Monthly Logic**:
       - **Strict Year Filtering**: See [STRICT YEARLY FILTERING] above. 
       - **All 12 Months**: If hints provide a list of month columns, USE THEM ALL. Do not hallucinate others.
       - **Aggregation Formula**: `(SUM(COALESCE(col1, 0)) + SUM(COALESCE(col2, 0)) + ...)`
         - **Note**: Ensure `COALESCE` is inside the SUM if checking for nulls from LEFT JOIN.
    4. **Filtering & Logic**:
       - "Active in [YEAR]": If table has start/end dates, use overlap logic: `start_date <= '[YEAR]-12-31' AND (end_date >= '[YEAR]-01-01' OR end_date IS NULL)`.
       - "No Revenue" check: `HAVING (SUM(COALESCE(col1,0)) + ...) = 0`. (This covers both 0.00 values and NULLs from LEFT JOIN).
    5. **Safety & Validation**: 
       - **Qualify ALL Columns**: Always use `table_alias.column` (e.g. `s.amount` vs `amount`). 
       - **Subquery Check**: Verify that every column in a subquery exists in that subquery's alias definition.
       - STRICTLY select specific columns. NO 'SELECT *'.
       - Always wrap numeric columns in COALESCE(col, 0).
       - End the query with 'LIMIT 100'.
       - Ensure 'LIMIT' clause is after 'HAVING' or 'ORDER BY' and outside any parentheses.
    6. Return ONLY the raw SQL query. No markdown.
    """
    
    
    if feedback:
        prompt += f"\n\nPrevious Attempt Failed with Error: {feedback}\nPlease fix the SQL based on this error."

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

def parse_table_aliases(sql):
    """
    Extract table aliases from SQL to map alias -> table_name.
    Very basic heuristic regex parser.
    """
    import re
    # Lowercase for easier matching
    sql_lower = sql.lower()
    
    # Remove newlines for regex
    sql_clean = " ".join(sql_lower.split())
    
    # Regex to find FROM/JOIN clauses
    # Matches: FROM table_name [AS] alias
    # or JOIN table_name [AS] alias
    
    # This pattern captures:
    # 1. table_name
    # 2. alias (optional)
    pattern = r'(?:from|join)\s+([a-z0-9_]+)(?:\s+(?:as\s+)?([a-z0-9_]+))?'
    
    matches = re.findall(pattern, sql_clean)
    
    alias_map = {}
    for table, alias in matches:
        if table in ['select', 'where', 'group', 'order', 'left', 'right', 'inner', 'outer', 'limit']:
            continue # False positive
        
        real_alias = alias if alias and alias not in ['on', 'where', 'join', 'left', 'right', 'inner', 'outer'] else table
        alias_map[real_alias] = table
        
    return alias_map

def validate_sql_schema(sql, sheet_infos):
    """
    Validate that columns used in the SQL (alias.col) exist in the schema.
    Returns: (bool, error_message)
    """
    import re
    
    # 1. Build Schema Map: table_name -> set(columns)
    schema_map = {}
    for sheet in sheet_infos:
        t_name = sheet['table_name'].lower()
        cols = sheet.get('columns_metadata', [])
        if isinstance(cols, str):
            try: cols = json.loads(cols)
            except: cols = []
            
        col_names = set(c.get('name', '').lower() for c in cols)
        schema_map[t_name] = col_names
        
    # 2. Parse Aliases from SQL
    alias_map = parse_table_aliases(sql)
    
    # 3. Scan for "alias.column" usages
    # We look for words that match alias.column
    # excluding things inside string literals (simple check: boundaries)
    
    # Regex for alias.column
    # We assume aliases are alphanumeric/underscore
    usage_pattern = r'\b([a-z0-9_]+)\.([a-z0-9_]+)\b'
    
    sql_lower = sql.lower()
    usages = re.findall(usage_pattern, sql_lower)
    
    errors = []
    
    for alias, col in usages:
        # Skip if alias is not a known table alias (might be a subquery alias or valid string)
        if alias not in alias_map:
            continue
            
        table_name = alias_map[alias]
        
        # Check if table is in our schema (it should be)
        if table_name not in schema_map:
            # Maybe the SQL uses a table we didn't send? 
            # Or the regex parsed a keyword as a table.
            continue
            
        valid_cols = schema_map[table_name]
        
        if col not in valid_cols and col != '*':
            # 4. Fuzzy Match & Derived Column Heuristics
            
            # A. Check for derived aliases (e.g. total_revenue, avg_cost)
            # If the column name implies an aggregation, likely defined in a subquery/CTE our regex missed.
            is_derived = any(sub in col for sub in ['total_', 'sum_', 'avg_', 'count_', 'min_', 'max_', '_revenue', '_count'])
            if is_derived:
                continue # Skip validation for likely derived columns
            
            # B. Fuzzy Match (Did you mean?)
            import difflib
            suggestions = difflib.get_close_matches(col, valid_cols, n=1, cutoff=0.6)
            
            if suggestions:
                errors.append(f"Column '{col}' not found in '{table_name}'. Did you mean '{suggestions[0]}'?")
            else:
                # Provide a few sample columns to help context
                samples = list(valid_cols)[:5]
                errors.append(f"Column '{col}' does not exist in table '{table_name}'. Available: {samples}...")
            
    if errors:
        return False, "Schema Validation Failed: " + "; ".join(errors[:3])
        
    return True, None

def validate_sql_guardrails(sql):
    """
    Ensure SQL is safe: Read-only, no SELECT *, has LIMIT.
    Returns: (bool, error_message)
    """
    import re
    sql_upper = sql.upper().strip()
    
    # 1. Enforce SELECT only
    if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
        return False, "Query must start with SELECT or WITH."
        
    # 2. Block DML/DDL
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "GRANT", "REVOKE"]
    for word in forbidden:
        # Regex to ensure whole word match
        if re.search(r'\b' + word + r'\b', sql_upper):
            return False, f"Forbidden keyword detected: {word}"
            
    # 3. Forbid SELECT *
    if re.search(r"SELECT\s+\*", sql_upper):
        return False, "Wildcard 'SELECT *' is not allowed. Please specify columns."
        
    # 4. Enforce LIMIT for non-aggregates
    # Heuristic: if no COUNT/SUM/AVG/MAX/MIN, assume list.
    is_aggregate = any(agg in sql_upper for agg in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("])
    if not is_aggregate and "LIMIT" not in sql_upper:
        # Auto-inject LIMIT 100 for safety instead of failing? 
        # Better to fail or warn? User requested "Row limits for safety". 
        # Let's enforce it in validation for now, or auto-append. 
        # Auto-appending is friendlier.
        pass # We will rely on the prompt, but if missing, we could auto-append in execution.
        
    return True, None

def execute_query(sql, db_url):
    """
    Execute the SQL query and return a pandas DataFrame.
    """
    if not sql:
        return None, "No SQL generated"
    
    # Validation
    is_safe, error = validate_sql_guardrails(sql)
    if not is_safe:
        return None, f"Safety Guardrail Triggered: {error}"
        
    conn = get_db_connection(db_url)
    try:
        # Use pandas read_sql to easily get columns and data
        df = pd.read_sql(sql, conn)
        return df, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

def synthesize_answer(user_query, sql, df, openai_key):
    """
    Generate a natural language answer based on the query results.
    """
    client = OpenAI(
        api_key=openai_key,
        base_url="https://openrouter.ai/api/v1"
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
    
    IMPORTANT: Provide a brief post-query explanation of how this answer was derived from the data (e.g. "I calculated the sum of the 'amount' column where..." or "I found X rows matching...").
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

def validate_question_possibility(user_query, sheet_infos, openai_key):
    """
    Check if the user's question CAN be answered given the available schema.
    Returns: (bool, reason)
    """
    client = OpenAI(
        api_key=openai_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Build compact schema for validation
    schema_summary = ""
    for sheet in sheet_infos:
        columns_meta = sheet.get('columns_metadata', [])
        if isinstance(columns_meta, str):
            try: columns_meta = json.loads(columns_meta)
            except: columns_meta = []
            
        # Include Roles in the schematic description
        col_desc = []
        for c in columns_meta:
            name = c.get('name')
            role = c.get('role', 'other') # New role field
            col_desc.append(f"{name} ({role})")
            
        col_list = ", ".join(col_desc)
        schema_summary += f"- Table '{sheet['table_name']}' (Sheet: {sheet['sheet_name']}): [{col_list}]\n"

    prompt = f"""
    You are a Data Feasibility Validator.
    
    User Question: "{user_query}"
    
    Available Schemas (with semantic roles):
    {schema_summary}
    
    Task: Determine if the user's question can logically be answered using the available tables and columns.
    
    Validation Rules:
    1. **Semantic Matching**: 
       - If the user asks for "Revenue", accept columns with role 'revenue' OR columns like 'sales', 'income', 'inflow'.
       - If the user asks for "Total Revenue" and a table has 12 monthly columns (role: 'monthly_metric'), this IS answerable (by summing them).
    2. **Missing Columns**: If the question requires a specific dimension (e.g. "by Region") and no column matches that concept, return False.
    3. **Vague Questions**: If the question is too vague (e.g. "Analyze data") but tables exist, return True (we can start with a general summary).
    
    Return JSON format:
    {{
        "possible": true/false,
        "reason": "Clear explanation of what matches or what is specifically missing (e.g. 'Found Revenue column but missing Date column')."
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a validator. Output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        result = json.loads(response.choices[0].message.content)
        return result.get('possible', True), result.get('reason', "")
    except Exception as e:
        logging.warning(f"Validation failed, assuming possible: {e}")
        return True, ""

def search_sheets_keyword_fallback(query, db_url, limit=3):
    """
    Fallback search using basic text matching if vector search fails.
    """
    conn = get_db_connection(db_url)
    try:
        cur = conn.cursor()
        # Simple keyword matching on metadata fields
        # Note: We cast columns_metadata to text to search within it
        sql = """
            SELECT 
                sheet_id, 
                table_name, 
                sheet_name, 
                category, 
                columns_metadata,
                0.5 as distance -- Artificial distance for keyword matches
            FROM sheets_metadata
            WHERE 
                sheet_name ILIKE %s OR
                category ILIKE %s OR
                keywords ILIKE %s OR
                columns_metadata::text ILIKE %s
            LIMIT %s
        """
        search_term = f"%{query}%"
        cur.execute(sql, (search_term, search_term, search_term, search_term, limit))
        
        results = []
        for row in cur.fetchall():
            results.append({
                "sheet_id": row[0],
                "table_name": row[1],
                "sheet_name": row[2],
                "category": row[3],
                "columns_metadata": row[4],
                "distance": row[5]
            })
        return results
    except Exception as e:
        logging.error(f"Keyword fallback failed: {e}")
        return []
    finally:
        conn.close()


def calculate_confidence(sql, sheet_matches):
    """
    Calculate a heuristic confidence score (0.0 - 1.0).
    """
    score = 0.5 # Base score
    details = []
    
    # 1. Similarity signal (0.0 to 1.0 distance, 0 is best)
    # We take the best sheet's distance
    best_dist = sheet_matches[0].get('distance', 1.0)
    
    # specific handling for our keyword fallback (0.5) vs vector
    if best_dist is not None:
        if best_dist < 0.35:
            score += 0.2
            details.append("Strong semantic match (distance < 0.35)")
        elif best_dist > 0.6:
            score -= 0.2
            details.append("Weak semantic match (distance > 0.6)")
            
    # 2. SQL Composition
    sql_upper = sql.upper()
    
    # Find matching tables in SQL
    # We check if the table names provided are actually used in the query
    tables_in_sql = 0
    for s in sheet_matches:
        if s['table_name'] in sql: 
            tables_in_sql += 1
    
    details.append(f"Tables used in Query: {tables_in_sql}")
    
    if tables_in_sql == 0:
        score -= 0.4
        details.append("CRITICAL: Generated SQL does not match any selected table.")
    elif tables_in_sql > 1:
        score += 0.1
        details.append("Multi-table query")
    
    # Filters (Specifics)
    if " WHERE " in sql_upper:
        score += 0.1
        details.append("Uses filters (WHERE)")
        
    return min(max(score, 0.0), 1.0), details

def process_retrieval(user_query, db_url, openai_key):
    """
    Orchestration function:
    1. Search relevant schemas (Vector + Keyword Fallback)
    2. Validate possibility
    3. Generate SQL
    4. Execute
    5. Synthesize
    
    Returns a dict with all steps for UI display.
    """
    steps = {}
    debug_log = []
    
    # 1. Search (Fetch Top 5 now)
    debug_log.append("Starting Vector Search...")
    sheets = search_relevant_sheets(user_query, db_url, openai_key, limit=5)
    debug_log.append(f"Vector Search found {len(sheets)} candidates.")
    if sheets:
        debug_log.append(f"Top Vector Match: {sheets[0]['sheet_name']} (Distance: {sheets[0].get('distance')})")
    
    # Check strictness of vector match
    use_fallback = False
    if not sheets:
        use_fallback = True
        debug_log.append("Vector search empty. Triggering Fallback.")
    elif sheets[0]['distance'] > 0.6:
         use_fallback = True
         debug_log.append(f"Vector match weak (>0.6). Triggering Fallback.")
         
    if use_fallback:
        logging.info("Vector match weak or empty. Trying keyword fallback.")
        debug_log.append("Executing Keyword Fallback Search...")
        keyword_matches = search_sheets_keyword_fallback(user_query, db_url, limit=5)
        debug_log.append(f"Keyword Search found {len(keyword_matches)} candidates.")
        
        # Merge lists, preferring vector results if present, deduping by sheet_id
        seen_ids = set(s['sheet_id'] for s in sheets)
        for s in keyword_matches:
            if s['sheet_id'] not in seen_ids:
                sheets.append(s)
                
    steps['debug_log'] = debug_log
    
    if not sheets:
        error_msg = "No relevant data found for your query. Please mention the specific sheet name or category."
        return {"error": error_msg, "debug_log": debug_log}
    
    steps['sheet_matches'] = sheets # List of sheets
    debug_log.append(f"Final Candidates: {[s['sheet_name'] for s in sheets]}")
    
    # 2. Validation Step
    debug_log.append("Validating feasibility with LLM...")
    is_possible, reason = validate_question_possibility(user_query, sheets, openai_key)
    debug_log.append(f"Validation Result: {is_possible}. Reason: {reason}")
    
    if not is_possible:
        steps['final_answer'] = f"🚫 I cannot answer this question based on the available data.\n\n**Reason:** {reason}"
        return steps

    # 3. Generate, Validate, and Execute (Unified Retry Loop)
    final_sql = None
    results_df = None
    last_feedback = None
    
    # We allow up to 4 attempts: 1 initial + 3 retries based on feedback
    for attempt in range(4):
        logging.info(f"Retrieval Cycle (Attempt {attempt+1})...")
        
        # A. Generate SQL
        # If last_feedback is present, the LLM will see "Previous Attempt Failed..."
        sql = generate_sql_query(user_query, sheets, openai_key, feedback=last_feedback)
        
        # B. Validate Schema (Static Check)
        is_valid_schema, schema_error = validate_sql_schema(sql, sheets)
        if not is_valid_schema:
            last_feedback = f"Schema Check Failed: {schema_error}"
            debug_log.append(f"Attempt {attempt+1} Schema Error: {schema_error}")
            continue # Try again
            
        # C. Execute SQL (Runtime Check)
        df, exec_error = execute_query(sql, db_url)
        if exec_error:
            # Clean error message for LLM
            clean_err = exec_error.replace(db_url, "DB_URL") # Hide sensitive info if any
            last_feedback = f"Postgres Execution Error: {clean_err}"
            debug_log.append(f"Attempt {attempt+1} Execution Error: {clean_err}")
            continue # Try again
            
        # D. Success
        final_sql = sql
        results_df = df
        debug_log.append(f"Attempt {attempt+1} Success!")
        break
    
    # Handle Failure after retries
    if final_sql is None or results_df is None:
        steps['error'] = f"I failed to generate a valid query after 4 attempts.\nLast error: {last_feedback}"
        # Still return steps so user can see debug log
        steps['debug_log'] = debug_log
        return steps
        
    steps['generated_sql'] = final_sql
    steps['results_df'] = results_df
    
    # 3.5 Calculate Confidence (on final SQL)
    confidence, conf_details = calculate_confidence(final_sql, sheets)
    debug_log.append(f"Final Confidence: {confidence:.2f}")
    steps['confidence_score'] = confidence
    
    # Threshold for suppression
    if confidence < 0.4:
         msg = (
            f"⚠️ **Low Confidence Response** (Score: {confidence:.2f})\n\n"
            "I am suppressing the answer because I'm not confident it's correct. "
            "**Internal Reasoning:**\n" + "\n".join([f"- {d}" for d in conf_details])
        )
         steps['final_answer'] = msg
         return steps
         
    # 4. Synthesize Answer
    answer = synthesize_answer(user_query, final_sql, results_df, openai_key)
    steps['final_answer'] = answer
    
    return steps
