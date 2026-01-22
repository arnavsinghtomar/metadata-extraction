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

def generate_fast_analysis(df, sheet_name, table_name):
    """
    Fast, heuristic-based metadata generation to replace slow LLM calls.
    Returns dict with category, summary, keywords.
    """
    # 1. Category Heuristics
    category = "General"
    cols_lower = [str(c).lower() for c in df.columns]
    name_lower = str(sheet_name).lower()
    
    financial_terms = ['revenue', 'sales', 'profit', 'forecast', 'budget', 'p&l', 'balance', 'cost', 'expense', 'margin', 'fiscal']
    hr_terms = ['employee', 'staff', 'hr', 'salary', 'payroll', 'hiring', 'recruiting']
    inventory_terms = ['stock', 'inventory', 'sku', 'warehouse', 'product', 'item']
    
    if any(x in name_lower or any(x in c for c in cols_lower) for x in financial_terms):
        category = "Financial"
    elif any(x in name_lower or any(x in c for c in cols_lower) for x in hr_terms):
        category = "HR"
    elif any(x in name_lower or any(x in c for c in cols_lower) for x in inventory_terms):
        category = "Inventory"
        
    # 2. Detailed Summary
    # "Financial data table 'Sheet1' containing: Date, Amount..."
    key_cols = df.columns.tolist()[:10] # Top 10 cols
    cols_str = ", ".join([str(c) for c in key_cols])
    if len(df.columns) > 10:
        cols_str += "..."
        
    summary = f"{category} data table '{sheet_name}' (DB: {table_name}). structure includes columns: {cols_str}. Contains {len(df)} rows."
    
    # 3. Keywords
    # Combine sheet name, category, and clean column names
    clean_cols = [str(c) for c in key_cols if len(str(c)) > 2 and 'unnamed' not in str(c).lower()]
    keywords = f"{sheet_name}, {category}, {', '.join(clean_cols)}"
    
    return {
        "category": category,
        "summary": summary,
        "keywords": keywords
    }

def get_embeddings_batch(texts, openai_key):
    """
    Generate multiple embeddings in a single API call.
    Returns a list of vectors.
    """
    if not texts or not openai_key:
        return [None] * len(texts)
    
    # Filter out empty strings but keep track of indices to return correct size
    valid_indices = [i for i, t in enumerate(texts) if t]
    valid_texts = [texts[i] for i in valid_indices]
    
    if not valid_texts:
        return [None] * len(texts)

    try:
        client = OpenAI(
            api_key=openai_key,
            base_url="https://openrouter.ai/api/v1"
        )
        response = client.embeddings.create(
            input=valid_texts,
            model="openai/text-embedding-3-small"
        )
        
        # Map back to original order
        # Response data is ordered by input
        results = [None] * len(texts)
        for idx, embed_data in enumerate(response.data):
            original_idx = valid_indices[idx]
            results[original_idx] = embed_data.embedding
            
        return results
    except Exception as e:
        logging.error(f"Failed to generate batch embeddings: {e}")
        # Fallback to None
        return [None] * len(texts)

def get_embedding(text, openai_key):
    """Wrapper for single embedding (backward compatibility)"""
    res = get_embeddings_batch([text], openai_key)
    return res[0]

def generate_fast_file_analysis(file_name, sheet_summaries):
    """
    Fast heuristic for file-level summary.
    """
    if not sheet_summaries:
        return {"summary": f"File '{file_name}' with no readable data.", "keywords": file_name}
        
    categories = set(s['category'] for s in sheet_summaries)
    sheet_names = [s['name'] for s in sheet_summaries]
    
    # Summary
    summary = (
        f"Excel file '{file_name}' containing {len(sheet_summaries)} tables. "
        f"Categories detected: {', '.join(categories)}. "
        f"Sheets/Tables include: {', '.join(sheet_names[:5])}"
    )
    if len(sheet_names) > 5:
        summary += ", ..."
        
    # Keywords
    keywords = f"{file_name}, {', '.join(categories)}, {', '.join(sheet_names[:5])}"
    
    return {"summary": summary, "keywords": keywords}

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
    
    if not name:
        return "unnamed"
        
    # Ensure it doesn't start with a digit
    if name[0].isdigit():
        name = f"col_{name}"
        
    return name

def detect_tables(df):
    """
    Fast Vectorized Table Detection.
    Splits DataFrame based on fully empty rows and columns.
    Complexity: O(R + C) instead of O(R*C).
    """
    if df.empty:
        return []

    # 1. Boolean mask: True where data exists
    # Treat whitespace as empty
    # optimizing: check for notna first (fast), then string check only on object cols
    mask = df.notna()
    
    # For object columns, ensure simple whitespace is False
    # This can be slow on huge frames, so we do a quick check
    if (mask.values.any()): # only check if there is any data
        for col in df.select_dtypes(include=['object']):
            # Update mask for this col: False if it is just whitespace
            # .str.strip().str.len() > 0 is vectorised
            mask[col] = mask[col] & (df[col].astype(str).str.strip().str.len() > 0)

    if not mask.values.any():
        return []
        
    # 2. Find Empty Rows (vectorized)
    # A row is "has_data" if any column is True
    rows_with_data = mask.any(axis=1)
    
    # 3. Identify Row Blocks
    # Logic: Group consecutive True values in rows_with_data
    # We use a trick: diff() != 0 marks boundaries
    # We filter only indices where rows_with_data is True
    
    tables = []
    
    # Get integer indices of rows that have data
    valid_row_indices = rows_with_data[rows_with_data].index.tolist()
    
    if not valid_row_indices:
        return []
        
    # Group consecutive indices
    from itertools import groupby
    from operator import itemgetter
    
    row_groups = []
    for k, g in groupby(enumerate(valid_row_indices), lambda ix: ix[0] - ix[1]):
        row_groups.append(list(map(itemgetter(1), g)))
        
    # 4. For each Row Block, Split by Columns
    for r_group in row_groups:
        start_r = r_group[0]
        end_r = r_group[-1]
        
        # Horizontal slice
        row_slice = df.iloc[start_r : end_r + 1]
        slice_mask = mask.iloc[start_r : end_r + 1]
        
        # Check for empty columns in this slice
        cols_with_data = slice_mask.any(axis=0)
        valid_col_indices = cols_with_data[cols_with_data].index.tolist() # These are column NAMES, not int indices usually
        
        # Map to integer locations for slicing
        # df.columns.get_loc can be slow if called many times, better to use integer index
        col_int_indices = [df.columns.get_loc(c) for c in valid_col_indices]
        col_int_indices.sort()
        
        if not col_int_indices:
            continue
            
        # Group consecutive column indices
        col_groups = []
        for k, g in groupby(enumerate(col_int_indices), lambda ix: ix[0] - ix[1]):
            col_groups.append(list(map(itemgetter(1), g)))
            
        for c_group in col_groups:
            start_c = c_group[0]
            end_c = c_group[-1]
            
            # Extract Table
            sub_df = row_slice.iloc[:, start_c : end_c + 1]
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
        
        # We will collect everything in memory first to batch API calls
        # Structure: list of dicts with all info needed for DB insert
        tables_buffer = [] 
        # Structure: list of strings to embed. Order matters!
        # We will map them back using indices.
        embedding_payloads = [] 
        
        logging.info(f"Parsing {len(sheet_names)} sheets for {file_name}...")
        
        for sheet_name in sheet_names:
            # Load sheet (header=None for detection)
            raw_df = xl.parse(sheet_name, header=None)
            sub_tables_df = detect_tables(raw_df)
            
            if not sub_tables_df:
                continue
                
            for sub_i, sub_df in enumerate(sub_tables_df):
                # 1. Clean and Header Promotion
                sub_df = sub_df.reset_index(drop=True)
                new_header = sub_df.iloc[0]
                sub_df = sub_df[1:]
                sub_df.columns = new_header
                sub_df.reset_index(drop=True, inplace=True)
                
                # Cleanup
                sub_df.dropna(how='all', axis=0, inplace=True)
                sub_df.dropna(how='all', axis=1, inplace=True)
                if sub_df.empty or sub_df.shape[1] == 0:
                    continue
                
                # Naming
                clean_sheet_name = sheet_name
                if len(sub_tables_df) > 1:
                    clean_sheet_name = f"{sheet_name}_Table{sub_i+1}"
                
                # Normalization
                df = sub_df
                original_cols = df.columns
                new_cols = [normalize_name(c) for c in original_cols]
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
                
                # Schema Inference
                column_defs = []
                column_metadata_list = []
                for idx, col in enumerate(df.columns):
                    pg_type = infer_pg_type(df[col])
                    column_defs.append(f"{col} {pg_type}")
                    try:
                        samples = df[col].dropna().unique().tolist()[:3]
                        samples = [str(s) for s in samples]
                    except:
                        samples = []
                    role = infer_column_role(col, pg_type)
                    column_metadata_list.append({
                        "name": col,
                        "original_name": str(original_cols[idx]),
                        "type": pg_type,
                        "role": role,
                        "samples": samples
                    })
                
                # Table SQL
                norm_sheet_name = normalize_name(clean_sheet_name)
                short_id = str(file_id).replace('-', '')[:8]
                table_name = f"sheet_{norm_sheet_name}_{short_id}"
                col_def_str = ",\n    ".join(column_defs)
                create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n    {col_def_str}\n);"
                
                # Metadata (Fast Heuristic)
                ai_data = generate_fast_analysis(df, clean_sheet_name, table_name)
                summary = ai_data.get('summary')
                category = ai_data.get('category')
                keywords = ai_data.get('keywords')
                
                sheet_summaries_list.append({
                    "name": clean_sheet_name,
                    "category": category,
                    "summary": summary
                })
                
                # Prepare Embedding Texts
                columns_text = f"Sheet: {clean_sheet_name}\n"
                for col_meta in column_metadata_list:
                    c_name = col_meta['name']
                    c_orig = col_meta['original_name']
                    c_samples = ", ".join(col_meta['samples'])
                    columns_text += f"Column: {c_name} (Original: {c_orig}) - Samples: {c_samples}\n"
                
                # Register payload for batching
                # Order: [Summary, Keywords, Columns]
                base_idx = len(embedding_payloads)
                embedding_payloads.extend([str(summary), str(keywords), str(columns_text)])
                
                # Store buffer
                tables_buffer.append({
                    "df": df,
                    "table_name": table_name,
                    "create_sql": create_table_sql,
                    "sheet_name": clean_sheet_name,
                    "num_rows": len(df),
                    "num_cols": len(df.columns),
                    "category": category,
                    "summary": summary,
                    "keywords": keywords,
                    "column_metadata_list": column_metadata_list,
                    "embed_indices": (base_idx, base_idx+1, base_idx+2) # Indices in the master payload list
                })

        # 2. File Level Metadata (Fast Heuristic)
        file_ai = generate_fast_file_analysis(file_name, sheet_summaries_list)
        f_summary = file_ai.get('summary')
        f_keywords = file_ai.get('keywords')
        
        # Add file embeddings to payload
        file_embed_base_idx = len(embedding_payloads)
        embedding_payloads.extend([str(f_summary), str(f_keywords)])
        
        # 3. Batch Call for Embeddings (SINGLE API CALL)
        logging.info(f"Generating {len(embedding_payloads)} embeddings in batch for {file_name}...")
        all_embeddings = get_embeddings_batch(embedding_payloads, openai_key)
        
        # 4. Database Transactions
        with conn.cursor() as cur:
            # A. Insert File Metadata
            f_summ_emb = all_embeddings[file_embed_base_idx]
            f_key_emb = all_embeddings[file_embed_base_idx+1]
            
            cur.execute(
                """
                INSERT INTO files_metadata (file_id, file_name, num_sheets, summary, keywords, summary_embedding, keywords_embedding) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (str(file_id), file_name, len(sheet_names), f_summary, f_keywords, f_summ_emb, f_key_emb)
            )
            
            # B. Process Tables
            for tbl in tables_buffer:
                # Retrieve embeddings
                idx_s, idx_k, idx_c = tbl['embed_indices']
                s_emb = all_embeddings[idx_s]
                k_emb = all_embeddings[idx_k]
                c_emb = all_embeddings[idx_c]
                
                # Create Table
                cur.execute(tbl['create_sql'])
                
                # Insert Data
                df = tbl['df']
                df_to_insert = df.where(pd.notnull(df), None)
                columns_list = ", ".join(df.columns)
                placeholders = ", ".join(["%s"] * len(df.columns))
                insert_query = f"INSERT INTO {tbl['table_name']} ({columns_list}) VALUES ({placeholders})"
                cur.executemany(insert_query, df_to_insert.values.tolist())
                
                # Insert Sheet Metadata
                sheet_id = uuid.uuid4()
                cur.execute(
                    """
                    INSERT INTO sheets_metadata 
                    (sheet_id, file_id, sheet_name, table_name, num_rows, num_columns, 
                     category, summary, keywords, summary_embedding, keywords_embedding, columns_embedding, columns_metadata) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (str(sheet_id), str(file_id), tbl['sheet_name'], tbl['table_name'], tbl['num_rows'], tbl['num_cols'],
                     tbl['category'], tbl['summary'], tbl['keywords'], s_emb, k_emb, c_emb, json.dumps(tbl['column_metadata_list']))
                )

        conn.commit()
        logging.info(f"Successfully ingested {file_name} with {len(tables_buffer)} tables.")
        
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
