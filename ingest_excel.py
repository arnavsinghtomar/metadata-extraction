import os
import sys
import uuid
import re
import logging
import json
import pandas as pd
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    # Use OpenRouter Key
    openai_key = os.getenv("OPENROUTER_API_KEY")
    
    if not db_url:
        logging.error("DATABASE_URL not found in .env environment.")
        sys.exit(1)
    if not openai_key:
        logging.warning("OPENROUTER_API_KEY not found. AI features will be skipped.")
    
    return db_url, openai_key

def generate_ai_analysis(df, sheet_name, table_name, openai_key):
    """
    Generate category, keywords, and summary using OpenRouter (OpenAI models).
    Returns a dict.
    """
    if not openai_key:
        return None
        
    try:
        client = OpenAI(
            api_key=openai_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Create a compact representation of the data
        sample_data = df.head(5).to_markdown(index=False)
        columns = ", ".join(df.columns.tolist())
        rows, cols = df.shape
        
        prompt = (
            f"Analyze the Excel sheet named '{sheet_name}'.\n"
            f"Columns: {columns}\n"
            f"Shape: {rows} rows, {cols} columns.\n"
            f"Sample Data:\n{sample_data}\n\n"
            f"Return a JSON object with three keys:\n"
            f"1. 'category': A short category name (e.g., 'Financial', 'Inventory', 'HR', 'Sales').\n"
            f"2. 'summary': A concise 1-2 sentence summary.\n"
            f"3. 'keywords': A list of 5-10 identifying keywords or phrases as a single string separated by commas."
        )
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a data analyst. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        logging.error(f"Failed to generate AI analysis for {sheet_name}: {e}")
        raise e

def get_embedding(text, openai_key):
    """Generate vector embedding for text using OpenRouter."""
    if not text or not openai_key:
        return None
    try:
        client = OpenAI(
            api_key=openai_key,
            base_url="https://openrouter.ai/api/v1"
        )
        response = client.embeddings.create(
            input=text,
            model="openai/text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to generate embedding: {e}")
        raise e

def generate_file_level_analysis(file_name, sheet_summaries, openai_key):
    """
    Generate a file-level summary and keywords based on sheet summaries.
    """
    if not openai_key or not sheet_summaries:
        return None
        
    try:
        client = OpenAI(
            api_key=openai_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        summaries_text = "\n".join([f"- {s['name']} ({s['category']}): {s['summary']}" for s in sheet_summaries])
        
        prompt = (
            f"Analyze the Excel file named '{file_name}' based on the following summaries of its sheets:\n\n"
            f"{summaries_text}\n\n"
            f"Return a JSON object with two keys:\n"
            f"1. 'summary': A comprehensive 2-3 sentence summary of the entire file's purpose and content.\n"
            f"2. 'keywords': A list of 5-10 high-level keywords describe the entire file, as a single comma-separated string."
        )
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a data analyst. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Failed to generate file-level analysis: {e}")
        return None

def get_db_connection(db_url):
    """Connect to the Postgres database."""
    try:
        conn = psycopg2.connect(db_url)
        # Try to register pgvector type handler, but don't fail if extension doesn't exist yet
        try:
            register_vector(conn)
        except Exception as e:
            # This is expected if 'vector' extension is not installed yet
            logging.warning(f"Could not register vector type: {e}. Expect this on first run.")
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        sys.exit(1)

def normalize_name(name):
    """
    Normalize string to be used as table or column name.
    """
    if not name:
        return "unnamed"
    
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    
    return name if name else "unnamed"

def detect_tables(df):
    """
    Detects multiple distinct tables within a single DataFrame based on empty rows and columns.
    Returns a list of DataFrames.
    """
    # 1. Binarize: True if cell has data, False if empty/NaN
    mask = df.notna() & df.apply(lambda x: x.astype(str).str.strip() != "")
    
    # If the sheet is empty, return empty list
    if not mask.any().any():
        return []

    # 2. Find connected components
    rows, cols = df.shape
    visited = set()
    tables = []

    # 8-connectivity to handle diagonal touches
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]

    for r in range(rows):
        for c in range(cols):
            if mask.iat[r, c] and (r, c) not in visited:
                # Start a new component
                stack = [(r, c)]
                visited.add((r, c))
                
                min_r, max_r = r, r
                min_c, max_c = c, c
                
                while stack:
                    curr_r, curr_c = stack.pop()
                    
                    min_r = min(min_r, curr_r)
                    max_r = max(max_r, curr_r)
                    min_c = min(min_c, curr_c)
                    max_c = max(max_c, curr_c)
                    
                    for dr, dc in directions:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if mask.iat[nr, nc] and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                stack.append((nr, nc))
                
                # Extract the rectangular sub-dataframe
                sub_df = df.iloc[min_r : max_r + 1, min_c : max_c + 1]
                
                # Check minimum size to avoid noise (e.g., must have at least 2x2 or 1x3?)
                # For now, accept anything not trivial (e.g. > 1 cell)
                if sub_df.size > 1:
                     tables.append(sub_df)

    return tables

def infer_pg_type(series):
    """
    Infer Postgres column type from pandas Series.
    """
    # Check for mixed types or object types that might be specific
    if isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(series):
        return "TEXT"
        
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
        
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
        
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE PRECISION"
        
    
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    
    return "TEXT"


def infer_column_role(name, dtype):
    """
    Infer the semantic role of a column based on name and type.
    """
    name = name.lower()
    
    # Date/Time
    if dtype == 'TIMESTAMP' or any(x in name for x in ['date', 'time', 'period', 'year', 'month', 'day', 'timestamp']):
        return 'date'
    
    # Numeric checks
    if dtype in ['DOUBLE PRECISION', 'INTEGER']:
        # Revenue synonyms
        if any(x in name for x in ['revenue', 'income', 'sales', 'billing', 'turnover', 'inflow', 'amount', 'credit', 'receivable']):
            return 'revenue'
        # Expense synonyms
        if any(x in name for x in ['expense', 'cost', 'spending', 'outflow', 'debit', 'payable', 'fee', 'charge']):
            return 'expense'
        # Monthly data (e.g. jan_2024)
        months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        if any(m in name for m in months):
            return 'monthly_metric'
        # IDs that are numbers
        if any(x in name for x in ['id', 'code', 'num_']):
            return 'identifier'
            
    # Text identifiers
    if any(x in name for x in ['id', 'code', 'ref', 'name', 'vendor', 'customer', 'client', 'project', 'sku']):
        return 'identifier'
        
    return 'other'


def create_metadata_tables(conn):
    """Create the required metadata tables and extensions."""
    
    commands = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        """
        CREATE TABLE IF NOT EXISTS files_metadata (
            file_id UUID PRIMARY KEY,
            file_name TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT NOW(),
            num_sheets INTEGER,
            summary TEXT,
            keywords TEXT,
            summary_embedding vector(1536),
            keywords_embedding vector(1536)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS sheets_metadata (
            sheet_id UUID PRIMARY KEY,
            file_id UUID REFERENCES files_metadata(file_id),
            sheet_name TEXT,
            table_name TEXT,
            num_rows INTEGER,
            num_columns INTEGER,
            category TEXT,
            summary TEXT,
            keywords TEXT,
            summary_embedding vector(1536),
            keywords_embedding vector(1536),
            columns_embedding vector(1536),
            columns_metadata JSONB
        )
        """
    ]
    
    try:
        with conn.cursor() as cur:
            for command in commands:
                cur.execute(command)
                
            # Perform migrations on existing tables if needed
            # 1. files_metadata migration
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='files_metadata'")
            file_cols = {row[0] for row in cur.fetchall()}
            
            file_alters = []
            if 'summary' not in file_cols: file_alters.append("ALTER TABLE files_metadata ADD COLUMN summary TEXT")
            if 'keywords' not in file_cols: file_alters.append("ALTER TABLE files_metadata ADD COLUMN keywords TEXT")
            if 'summary_embedding' not in file_cols: file_alters.append("ALTER TABLE files_metadata ADD COLUMN summary_embedding vector(1536)")
            if 'keywords_embedding' not in file_cols: file_alters.append("ALTER TABLE files_metadata ADD COLUMN keywords_embedding vector(1536)")
            
            for cmd in file_alters:
                cur.execute(cmd)
                logging.info(f"Executed file migration: {cmd}")

            # 2. sheets_metadata migration
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='sheets_metadata'")
            sheet_cols = {row[0] for row in cur.fetchall()}
            
            sheet_alters = []
            if 'category' not in sheet_cols: sheet_alters.append("ALTER TABLE sheets_metadata ADD COLUMN category TEXT")
            if 'summary' not in sheet_cols: sheet_alters.append("ALTER TABLE sheets_metadata ADD COLUMN summary TEXT")
            if 'keywords' not in sheet_cols: sheet_alters.append("ALTER TABLE sheets_metadata ADD COLUMN keywords TEXT")
            if 'summary_embedding' not in sheet_cols: sheet_alters.append("ALTER TABLE sheets_metadata ADD COLUMN summary_embedding vector(1536)")
            if 'keywords_embedding' not in sheet_cols: sheet_alters.append("ALTER TABLE sheets_metadata ADD COLUMN keywords_embedding vector(1536)")
            if 'columns_embedding' not in sheet_cols: sheet_alters.append("ALTER TABLE sheets_metadata ADD COLUMN columns_embedding vector(1536)")
            if 'columns_metadata' not in sheet_cols: sheet_alters.append("ALTER TABLE sheets_metadata ADD COLUMN columns_metadata JSONB")
            
            for cmd in sheet_alters:
                cur.execute(cmd)
                logging.info(f"Executed sheet migration: {cmd}")
                
        conn.commit()
        logging.info("Metadata tables and extensions verified.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error creating/updating metadata tables: {e}")
        sys.exit(1)


def process_excel_file(file_path, db_url, openai_key):
    """
    Main logic to ingest Excel file.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
        
    conn = get_db_connection(db_url)
    file_name = os.path.basename(file_path)
    file_id = uuid.uuid4()
    
    logging.info(f"Processing file: {file_name} (ID: {file_id})")
    
    sheet_summaries_list = [] # To store data for file-level summary
    
    try:
        # Load Excel file metadata first to count sheets
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        
        # 1. Insert file-level metadata (initial)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO files_metadata (file_id, file_name, num_sheets) VALUES (%s, %s, %s)",
                (str(file_id), file_name, len(sheet_names))
            )
        
        # 2. Process each sheet
        for sheet_name in sheet_names:
            logging.info(f"Processing sheet: {sheet_name}")
            
            # Load sheet with no header assumption initially
            # We want to detect blobs of data
            raw_df = xl.parse(sheet_name, header=None)
            
            # Detect sub-tables
            sub_tables_df = detect_tables(raw_df)
            
            if not sub_tables_df:
                logging.warning(f"Sheet '{sheet_name}' is empty or has no detected tables. Skipping.")
                continue
                
            logging.info(f"Sheet '{sheet_name}': Detected {len(sub_tables_df)} tables.")
            
            # Process each sub-table
            for sub_i, sub_df in enumerate(sub_tables_df):
                # Promote first row as header for the sub-table
                # We assume the top row of the block is the header
                # Reset index to clean up
                sub_df = sub_df.reset_index(drop=True)
                
                # Set first row as header
                new_header = sub_df.iloc[0] # grab the first row for the header
                sub_df = sub_df[1:] # take the data less the header row
                sub_df.columns = new_header # set the header row as the df header
                sub_df.reset_index(drop=True, inplace=True)
                
                # Determine display name
                clean_sheet_name = sheet_name
                if len(sub_tables_df) > 1:
                    clean_sheet_name = f"{sheet_name}_Table{sub_i+1}"
                
                logging.info(f"Processing sub-table: {clean_sheet_name}")

                # Drop fully empty cols/rows within this component (just in case)
                sub_df.dropna(how='all', axis=0, inplace=True)
                sub_df.dropna(how='all', axis=1, inplace=True)

                if sub_df.empty or sub_df.shape[1] == 0:
                    continue

                df = sub_df # alias for existing logic compatibility
                
                # Normalize column names
                original_cols = df.columns
                new_cols = [normalize_name(c) for c in original_cols]
                
                # Deduplicate column names if necessary
                final_cols = []
                seen_cols = {}
                for col in new_cols:
                    if col in seen_cols:
                        seen_cols[col] += 1
                        final_cols.append(f"{col}_{seen_cols[col]}")
                    else:
                        seen_cols[col] = 0
                        final_cols.append(col)
                
                df.columns = final_cols
                
                # Infer Schema and Collect Metadata
                column_defs = []
                column_metadata_list = []
                
                for idx, col in enumerate(df.columns):
                    pg_type = infer_pg_type(df[col])
                    column_defs.append(f"{col} {pg_type}")
                    
                    # Capture metadata (original name, samples, role)
                    try:
                        # Get top 3 unique non-null values for context
                        samples = df[col].dropna().unique().tolist()[:3]
                        # Convert to string to ensure JSON serializability for things like Dates/Decimals
                        samples = [str(s) for s in samples]
                    except Exception:
                        samples = []
                    
                    # Infer Role
                    role = infer_column_role(col, pg_type)
                    
                    column_metadata_list.append({
                        "name": col,
                        "original_name": str(original_cols[idx]),
                        "type": pg_type,
                        "role": role,
                        "samples": samples
                    })
                
                # Generate Table Name
                norm_sheet_name = normalize_name(clean_sheet_name)
                short_id = str(file_id).replace('-', '')[:8]
                table_name = f"sheet_{norm_sheet_name}_{short_id}"
                
                # Generate CREATE TABLE statement
                col_def_str = ",\n    ".join(column_defs)
                create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {col_def_str}\n);"
                
                # Explicitly log SQL for visibility
                print(f"\n--- SQL for {table_name} ---")
                print(create_table_sql)
                print("------------------------------\n")
                
                # Generate AI Analysis (Summary, Category, Keywords)
                ai_data = generate_ai_analysis(df, clean_sheet_name, table_name, openai_key)
                
                summary = None
                category = None
                keywords = None
                summary_embedding = None
                keywords_embedding = None
                
                if ai_data:
                    category = ai_data.get('category')
                    summary = ai_data.get('summary')
                    keywords = ai_data.get('keywords')
                    
                    # Store for file-level analysis
                    sheet_summaries_list.append({
                        "name": clean_sheet_name,
                        "category": category,
                        "summary": summary
                    })
                    
                    logging.info(f"Analysis for {clean_sheet_name}: Category={category}, Keywords={keywords}")
                    
                    # Generate embeddings
                    summary_embedding = get_embedding(summary, openai_key)
                    keywords_embedding = get_embedding(keywords, openai_key)
                
                # Generate Columns Embedding (Columns + Samples text)
                # Create a rich text representation for semantic column matching
                columns_text = f"Sheet: {clean_sheet_name}\n"
                for col_meta in column_metadata_list:
                    c_name = col_meta['name']
                    c_orig = col_meta['original_name']
                    c_samples = ", ".join(col_meta['samples'])
                    columns_text += f"Column: {c_name} (Original: {c_orig}) - Samples: {c_samples}\n"
                
                columns_embedding = get_embedding(columns_text, openai_key)

                with conn.cursor() as cur:
                    # Create Table
                    cur.execute(create_table_sql)
                    
                    # Insert Data
                    df_to_insert = df.where(pd.notnull(df), None)
                    columns_list = ", ".join(df.columns)
                    placeholders = ", ".join(["%s"] * len(df.columns))
                    insert_query = f"INSERT INTO {table_name} ({columns_list}) VALUES ({placeholders})"
                    
                    cur.executemany(insert_query, df_to_insert.values.tolist())
                    
                    # Insert Metadata with Embeddings
                    sheet_id = uuid.uuid4()
                    cur.execute(
                        """
                        INSERT INTO sheets_metadata 
                        (sheet_id, file_id, sheet_name, table_name, num_rows, num_columns, 
                         category, summary, keywords, summary_embedding, keywords_embedding, columns_embedding, columns_metadata) 
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (str(sheet_id), str(file_id), clean_sheet_name, table_name, len(df), len(df.columns),
                         category, summary, keywords, summary_embedding, keywords_embedding, columns_embedding, json.dumps(column_metadata_list))
                    )
        
        # 3. Generate and Update File Level Metadata
        if sheet_summaries_list:
            logging.info("Generating file-level summary...")
            file_ai = generate_file_level_analysis(file_name, sheet_summaries_list, openai_key)
            if file_ai:
                f_summary = file_ai.get('summary')
                f_keywords = file_ai.get('keywords')
                f_summ_emb = get_embedding(f_summary, openai_key)
                f_key_emb = get_embedding(f_keywords, openai_key)
                
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE files_metadata 
                        SET summary=%s, keywords=%s, summary_embedding=%s, keywords_embedding=%s
                        WHERE file_id=%s
                        """,
                        (f_summary, f_keywords, f_summ_emb, f_key_emb, str(file_id))
                    )
                logging.info("File-level metadata updated.")

        # Commit all changes
        conn.commit()
        logging.info("Transaction committed successfully.")
        
    except Exception as e:
        conn.rollback()
        logging.error(f"An error occurred: {e}")
        logging.info("Transaction rolled back.")
        sys.exit(1)
    finally:
        conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_excel.py <path_to_excel_file>")
        sys.exit(1)
        
    excel_file = sys.argv[1]
    db_url_env, openai_key_env = load_environment()
    
    # Initial connection to ensure metadata tables exist
    connection = get_db_connection(db_url_env)
    create_metadata_tables(connection)
    connection.close()
    
    # Process the file
    process_excel_file(excel_file, db_url_env, openai_key_env)
