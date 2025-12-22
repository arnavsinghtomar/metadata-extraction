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
        sql = """
            SELECT 
                sheet_id, 
                table_name, 
                sheet_name, 
                category, 
                columns_metadata,
                (summary_embedding <=> %s::vector) as distance
            FROM sheets_metadata
            ORDER BY distance ASC
            LIMIT %s
        """
        cur.execute(sql, (embedding, limit))
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

def generate_sql_query(user_query, sheet_info, openai_key):
    """
    Generate SQL for a specific sheet given its schema.
    """
    client = OpenAI(api_key=openai_key)
    
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
        
    prompt = f"""
    You are a PostgreSQL expert assisting a user with a financial query.
    
    Target Data Dictionary:
    {schema_desc}
    
    User Question: "{user_query}"
    
    Goal: Write a valid PostgreSQL query to answer the question.
    
    Rules:
    1. Use ONLY the table and columns provided in the schema.
    2. Use ILIKE for flexible text matching (e.g. WHERE vendor ILIKE '%amazon%').
    3. If the user asks for a total, use SUM/COUNT.
    4. Return ONLY the raw SQL query. No markdown (```sql), no explanations.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

def synthesize_answer(user_query, sql, df, openai_key):
    """
    Generate a natural language answer based on the query results.
    """
    client = OpenAI(api_key=openai_key)
    
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
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content

def process_retrieval(user_query, db_url, openai_key):
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
    sheets = search_relevant_sheets(user_query, db_url, openai_key, limit=1)
    if not sheets:
        return {"error": "No relevant data found for your query."}
    
    best_sheet = sheets[0]
    steps['sheet_match'] = best_sheet
    
    # 2. Generate SQL
    sql = generate_sql_query(user_query, best_sheet, openai_key)
    steps['generated_sql'] = sql
    
    # 3. Execute
    df, error = execute_query(sql, db_url)
    if error:
        steps['error'] = f"SQL Execution Failed: {error}"
        return steps
        
    steps['results_df'] = df
    
    # 4. Synthesize
    answer = synthesize_answer(user_query, sql, df, openai_key)
    steps['final_answer'] = answer
    
    return steps
