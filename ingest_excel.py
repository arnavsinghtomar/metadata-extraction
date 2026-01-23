import os
import sys
import uuid
import re
import logging
import json
import pandas as pd
import psycopg2
import hashlib
from psycopg2 import sql
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector
import concurrent.futures
import threading
import psycopg2.pool

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global Connection Pool and Lock
pg_pool = None
_pool_lock = threading.Lock()

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    
    if not db_url:
        logging.error("DATABASE_URL not found in .env environment.")
        sys.exit(1)
    if not openrouter_key:
        logging.warning("OPENROUTER_API_KEY not found. AI features will be skipped.")
    
    return db_url, openrouter_key

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

def init_db_pool(db_url):
    """Initialize the threaded connection pool securely."""
    global pg_pool
    if pg_pool is None:
        with _pool_lock:
            if pg_pool is None:  # Double-check locking pattern
                try:
                    pg_pool = psycopg2.pool.ThreadedConnectionPool(1, 60, db_url)
                    logging.info("Database connection pool initialized.")
                except Exception as e:
                    logging.error(f"Failed to init DB pool: {e}")
                    sys.exit(1)

def get_pooled_connection():
    """Get a connection from the pool."""
    global pg_pool
    if pg_pool is None:
        raise Exception("DB Pool not initialized")
    return pg_pool.getconn()

def release_pooled_connection(conn):
    """Return connection to the pool."""
    global pg_pool
    if pg_pool and conn:
        pg_pool.putconn(conn)

def normalize_name(name):
    """
    Normalize string to be used as table or column name.
    Ensures valid PostgreSQL identifiers (starts with letter/_, no keywords).
    """
    if not name:
        return "unnamed"
    
    name = str(name).lower()
    name = re.sub(r'[^a-z0-9]', '_', name)
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    
    if not name:
        return "unnamed"
        
    # Ensure it starts with a letter or underscore
    if name[0].isdigit():
        name = f"_{name}"
        
    # Check for reserved keywords (basic list)
    reserved = {'order', 'user', 'group', 'table', 'limit', 'offset', 'select', 'where', 'from', 'create', 'insert', 'update', 'delete', 'drop', 'alter', 'primary', 'key', 'references'}
    if name in reserved:
        name = f"_{name}"
    
    return name

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

def generate_ai_analysis(df, sheet_name, table_name, openai_key):
    """
    Generate category, keywords, and summary using OpenAI.
    Returns a dict.
    """
    if not openai_key:
        return None
        
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_key
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
        return None

def get_embedding(text, openai_key):
    """Generate vector embedding for text."""
    if not text or not openai_key:
        return None
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_key
        )
        response = client.embeddings.create(
            input=text,
            model="google/gemini-embedding-001"
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to generate embedding: {e}")
        return None

def get_embeddings_in_parallel(text_map, openai_key):
    """
    Fetch multiple embeddings in parallel.
    text_map: dict of {key: text}
    Returns: dict of {key: embedding}
    """
    if not openai_key:
        return {k: None for k in text_map}

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(text_map)) as executor:
        future_to_key = {
            executor.submit(get_embedding, text, openai_key): key 
            for key, text in text_map.items() if text
        }
        
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                logging.error(f"Error fetching embedding for {key}: {e}")
                results[key] = None
    return results

def process_sub_table_task(df, file_id, clean_sheet_name, db_url, openai_key):
    """
    Worker function to process a single sub-table in a separate thread.
    Uses pooled connections and parallel API calls.
    """
    conn = None
    try:
        conn = get_pooled_connection()
        
        # Drop fully empty cols/rows within this component
        df.dropna(how='all', axis=0, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)

        if df.empty or df.shape[1] == 0:
            return None

        # Normalize column names
        original_cols = df.columns
        new_cols = [normalize_name(c) for c in original_cols]
        
        # Deduplicate column names
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
            
            # Capture metadata
            try:
                samples = df[col].dropna().unique().tolist()[:3]
                samples = [str(s) for s in samples]
            except Exception:
                samples = []
            
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
        
        # 1. Generate AI Analysis (This is 1 API Call)
        ai_data = generate_ai_analysis(df, clean_sheet_name, table_name, openai_key)
        
        summary = None
        category = None
        keywords = None
        
        if ai_data:
            category = ai_data.get('category')
            summary = ai_data.get('summary')
            keywords = ai_data.get('keywords')
            logging.info(f"Analysis for {clean_sheet_name}: Category={category}")

        # 2. Prepare Columns Text
        columns_text = f"Sheet: {clean_sheet_name}\n"
        for col_meta in column_metadata_list:
            c_name = col_meta['name']
            c_orig = col_meta['original_name']
            c_samples = ", ".join(col_meta['samples'])
            columns_text += f"Column: {c_name} (Original: {c_orig}) - Samples: {c_samples}\n"
        
        # 3. Fetch Embeddings in Parallel (Summary, Keywords, Columns)
        embedding_inputs = {
            "summary": summary,
            "keywords": keywords,
            "columns": columns_text
        }
        
        embeddings_map = get_embeddings_in_parallel(embedding_inputs, openai_key)
        
        summary_embedding = embeddings_map.get("summary")
        keywords_embedding = embeddings_map.get("keywords")
        # Add columns_embedding if logic requires (the previous code had it in one version, omitted in another)
        # Let's check create_metadata_tables - it does not seem to have columns_embedding in sheets_metadata? 
        # Wait, the Raghvender_tyagi version of create_metadata_tables DOES NOT have columns_embedding in sheets_metadata!
        # It has `columns_metadata JSONB` but NO `columns_embedding vector`.
        # However, it has `summary_embedding` and `keywords_embedding`.
        # So I will skip columns_embedding for now to match the schema.
        
        # Database Ops
        with conn.cursor() as cur:
            # Create Table
            cur.execute(create_table_sql)
            
            # Insert Data
            df_to_insert = df.where(pd.notnull(df), None)
            columns_list = ", ".join(df.columns)
            placeholders = ", ".join(["%s"] * len(df.columns))
            insert_query = f"INSERT INTO {table_name} ({columns_list}) VALUES ({placeholders})"
            
            cur.executemany(insert_query, df_to_insert.values.tolist())
            
            # Insert Metadata
            sheet_id = uuid.uuid4()
            # Note: The merged code uses `sheets_metadata` columns: 
            # (sheet_id, file_id, sheet_name, table_name, num_rows, num_columns, category, summary, keywords, summary_embedding, keywords_embedding, columns_metadata)
            
            cur.execute(
                """
                INSERT INTO sheets_metadata 
                (sheet_id, file_id, sheet_name, table_name, num_rows, num_columns, 
                 category, summary, keywords, summary_embedding, keywords_embedding, columns_metadata) 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (str(sheet_id), str(file_id), clean_sheet_name, table_name, len(df), len(df.columns),
                 category, summary, keywords, summary_embedding, keywords_embedding, json.dumps(column_metadata_list))
            )
        
        conn.commit()
        return {
            "name": clean_sheet_name,
            "category": category,
            "summary": summary
        }

    except Exception as e:
        if conn: conn.rollback()
        logging.error(f"Failed to process sub-table {clean_sheet_name}: {e}")
        return None
    finally:
        if conn: release_pooled_connection(conn)

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
            summary_embedding vector(3072),
            keywords_embedding vector(3072)
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
            summary_embedding vector(3072),
            keywords_embedding vector(3072),
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
            
            # Check and Update Vector Dimensions if they exist but are wrong
            def check_and_fix_vector_dim(table, column, target_dim):
                cur.execute(f"""
                    SELECT atttypmod 
                    FROM pg_attribute 
                    WHERE attrelid = %s::regclass 
                    AND attname = %s
                """, (table, column))
                res = cur.fetchone()
                if res and res[0] != -1:
                    # atttypmod for vector(n) is n.
                    if res[0] != target_dim:
                        logging.info(f"Dimension mismatch for {table}.{column}: expected {target_dim}, found {res[0]}. Altering...")
                        cur.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE vector({target_dim})")

            # 1. files_metadata migration
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='files_metadata'")
            file_cols = {row[0] for row in cur.fetchall()}
            
            if 'summary' not in file_cols: cur.execute("ALTER TABLE files_metadata ADD COLUMN summary TEXT")
            if 'keywords' not in file_cols: cur.execute("ALTER TABLE files_metadata ADD COLUMN keywords TEXT")
            if 'summary_embedding' not in file_cols: 
                cur.execute("ALTER TABLE files_metadata ADD COLUMN summary_embedding vector(3072)")
            else:
                check_and_fix_vector_dim('files_metadata', 'summary_embedding', 3072)

            if 'keywords_embedding' not in file_cols: 
                cur.execute("ALTER TABLE files_metadata ADD COLUMN keywords_embedding vector(3072)")
            else:
                check_and_fix_vector_dim('files_metadata', 'keywords_embedding', 3072)

            # 2. sheets_metadata migration
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='sheets_metadata'")
            sheet_cols = {row[0] for row in cur.fetchall()}
            
            if 'category' not in sheet_cols: cur.execute("ALTER TABLE sheets_metadata ADD COLUMN category TEXT")
            if 'summary' not in sheet_cols: cur.execute("ALTER TABLE sheets_metadata ADD COLUMN summary TEXT")
            if 'keywords' not in sheet_cols: cur.execute("ALTER TABLE sheets_metadata ADD COLUMN keywords TEXT")
            
            if 'summary_embedding' not in sheet_cols: 
                cur.execute("ALTER TABLE sheets_metadata ADD COLUMN summary_embedding vector(3072)")
            else:
                check_and_fix_vector_dim('sheets_metadata', 'summary_embedding', 3072)

            if 'keywords_embedding' not in sheet_cols: 
                cur.execute("ALTER TABLE sheets_metadata ADD COLUMN keywords_embedding vector(3072)")
            else:
                check_and_fix_vector_dim('sheets_metadata', 'keywords_embedding', 3072)

            if 'columns_metadata' not in sheet_cols: cur.execute("ALTER TABLE sheets_metadata ADD COLUMN columns_metadata JSONB")
            
            # 3. Versioning migration
            if 'version' not in file_cols: cur.execute("ALTER TABLE files_metadata ADD COLUMN version INTEGER DEFAULT 1")
            if 'file_hash' not in file_cols: cur.execute("ALTER TABLE files_metadata ADD COLUMN file_hash TEXT")

        conn.commit()
        logging.info("Metadata tables and extensions verified.")
    except Exception as e:
        conn.rollback()
        logging.error(f"Error creating/updating metadata tables: {e}")
        sys.exit(1)

def generate_file_level_analysis(file_name, sheet_summaries, openai_key):
    """
    Generate a file-level summary and keywords based on sheet summaries.
    """
    if not openai_key or not sheet_summaries:
        return None
        
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_key
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

def compute_file_hash(file_path):
    """Compute MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def process_excel_file(file_path, db_url, openai_key, original_filename=None):
    """
    Main logic to ingest Excel file.
    Uses Threaded Pool and optimized flow.
    Returns: "SUCCESS", "DUPLICATE", or raises Exception.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        sys.exit(1)
    
    # Initialize Pool if not already
    init_db_pool(db_url)
    
    conn = get_pooled_connection()
    # Use provided filename (from upload) or fallback to path (local script usage)
    file_name = original_filename if original_filename else os.path.basename(file_path)
    file_id = uuid.uuid4()
    
    logging.info(f"Processing file: {file_name} (ID: {file_id})")
    
    try:
        # 0. Versioning & Duplicate Check
        current_hash = compute_file_hash(file_path)
        new_version = 1
        
        with conn.cursor() as cur:
            cur.execute(
                "SELECT version, file_hash FROM files_metadata WHERE file_name = %s ORDER BY version DESC LIMIT 1", 
                (file_name,)
            )
            row = cur.fetchone()
            if row:
                last_version, last_hash = row
                if last_hash == current_hash:
                    logging.info(f"File {file_name} is identical to version {last_version}. Skipping.")
                    return "DUPLICATE"
                new_version = last_version + 1
                logging.info(f"File {file_name} has content changes. Creating version {new_version}.")

        # Load Excel file metadata
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        
        # 1. Insert file-level metadata
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO files_metadata (file_id, file_name, num_sheets, version, file_hash) 
                VALUES (%s, %s, %s, %s, %s)
                """,
                (str(file_id), file_name, len(sheet_names), new_version, current_hash)
            )
        conn.commit()
        
        # Release this connection early since we don't need it for the parallel part
        release_pooled_connection(conn)
        conn = None 

        # 2. Parallel Processing (Sheets + Tables)
        # We define the inner function here to capture the 'xl' object closure
        def process_sheet_wrapper(sheet_name):
            try:
                # Parse specific sheet
                logging.info(f"Parsing sheet: {sheet_name}")
                raw_df = xl.parse(sheet_name, header=None)
                sub_tables_df = detect_tables(raw_df)
                
                if not sub_tables_df:
                    return []
                
                local_results = []
                for sub_i, sub_df in enumerate(sub_tables_df):
                    # Prepare DF
                    sub_df = sub_df.reset_index(drop=True)
                    new_header = sub_df.iloc[0]
                    sub_df = sub_df[1:]
                    sub_df.columns = new_header
                    sub_df.reset_index(drop=True, inplace=True)
                    
                    clean_sheet_name = sheet_name
                    if len(sub_tables_df) > 1:
                        clean_sheet_name = f"{sheet_name}_Table{sub_i+1}"
                    
                    # Process synchronously in this thread
                    res = process_sub_table_task(sub_df, file_id, clean_sheet_name, db_url, openai_key)
                    if res:
                        local_results.append(res)
                return local_results
            except Exception as e:
                logging.error(f"Error processing sheet {sheet_name}: {e}")
                return []

        # Use ThreadPoolExecutor for Sheet-Level Parallelism
        # This handles Parsing (CPU/IO) + SubTable Processing (IO) in parallel
        sheet_summaries_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_sheet = {executor.submit(process_sheet_wrapper, s): s for s in sheet_names}
            
            for future in concurrent.futures.as_completed(future_to_sheet):
                results = future.result()
                sheet_summaries_list.extend(results)

        # 4. Generate and Update File Level Metadata
        # Need a fresh connection
        if sheet_summaries_list:
            logging.info("Generating file-level summary...")
            file_ai = generate_file_level_analysis(file_name, sheet_summaries_list, openai_key)
            if file_ai:
                f_summary = file_ai.get('summary')
                f_keywords = file_ai.get('keywords')
                f_summ_emb = get_embedding(f_summary, openai_key)
                f_key_emb = get_embedding(f_keywords, openai_key)
                
                meta_conn = get_pooled_connection()
                try:
                    with meta_conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE files_metadata 
                            SET summary=%s, keywords=%s, summary_embedding=%s, keywords_embedding=%s
                            WHERE file_id=%s
                            """,
                            (f_summary, f_keywords, f_summ_emb, f_key_emb, str(file_id))
                        )
                    meta_conn.commit()
                finally:
                    release_pooled_connection(meta_conn)
                logging.info("File-level metadata updated.")

    except Exception as e:
        logging.error(f"An error occurred during file orchestration: {e}")
        raise e
    finally:
        if conn: release_pooled_connection(conn)
        logging.info("File processing completed.")
    
    return "SUCCESS"

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
