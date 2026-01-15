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
            model="openai/text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Failed to generate embedding: {e}")
        return None

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
            
            # Load sheet
            df = xl.parse(sheet_name)
            
            # Drop fully empty rows and columns
            df.dropna(how='all', axis=0, inplace=True)
            df.dropna(how='all', axis=1, inplace=True)
            
            if df.empty or df.shape[1] == 0:
                logging.warning(f"Sheet '{sheet_name}' is empty after cleaning. Skipping.")
                continue
            
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
                
                # Capture metadata (original name, samples)
                try:
                    # Get top 3 unique non-null values for context
                    samples = df[col].dropna().unique().tolist()[:3]
                    # Convert to string to ensure JSON serializability for things like Dates/Decimals
                    samples = [str(s) for s in samples]
                except Exception:
                    samples = []
                
                column_metadata_list.append({
                    "name": col,
                    "original_name": str(original_cols[idx]),
                    "type": pg_type,
                    "samples": samples
                })
            
            # Generate Table Name
            norm_sheet_name = normalize_name(sheet_name)
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
            ai_data = generate_ai_analysis(df, sheet_name, table_name, openai_key)
            
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
                    "name": sheet_name,
                    "category": category,
                    "summary": summary
                })
                
                logging.info(f"Analysis for {sheet_name}: Category={category}, Keywords={keywords}")
                
                # Generate embeddings
                summary_embedding = get_embedding(summary, openai_key)
                keywords_embedding = get_embedding(keywords, openai_key)

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
                     category, summary, keywords, summary_embedding, keywords_embedding, columns_metadata) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (str(sheet_id), str(file_id), sheet_name, table_name, len(df), len(df.columns),
                     category, summary, keywords, summary_embedding, keywords_embedding, json.dumps(column_metadata_list))
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
