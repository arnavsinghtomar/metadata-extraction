
import os
import pandas as pd
import uuid
import json
import logging
from psycopg2 import sql
from ingest_common import get_db_connection, get_embeddings_batch

def normalize_name(name):
    """
    Sanitize strings for SQL identifiers.
    """
    return "".join(c if c.isalnum() else "_" for c in str(name)).lower().strip("_")

def infer_column_role(col_name, dtype):
    """
    Heuristic to guess if a column is a Metric, Dimension, or Date.
    """
    col = col_name.lower()
    if 'date' in col or 'year' in col or 'month' in col or 'time' in col:
        return 'dimension_time'
    if 'id' in col or 'code' in col:
        return 'dimension_key'
    if 'total' in col or 'sum' in col or 'revenue' in col or 'profit' in col or 'cost' in col:
        return 'metric'
    
    if 'int' in dtype or 'float' in dtype:
        return 'metric' # Default numeric to metric
    return 'dimension' # Default text to dimension

def generate_heuristics(df, sheet_name):
    """
    Generate fast metadata without LLM calls for speed.
    """
    # 1. Keywords
    cols = df.columns.tolist()
    keywords = [sheet_name] + [str(c) for c in cols[:5]]
    keywords_str = ", ".join(keywords)
    
    # 2. Summary
    col_str = ", ".join([str(c) for c in cols[:10]])
    summary = f"Structured data table '{sheet_name}' containing columns: {col_str}. {len(df)} rows."
    
    return summary, keywords_str

def process_structured_file(file_path, db_url, openai_key, original_filename=None):
    """
    Ingest CSV or Excel file.
    """
    file_path = os.path.abspath(file_path)
    file_name = original_filename if original_filename else os.path.basename(file_path)
    file_id = uuid.uuid4()
    ext = file_name.split('.')[-1].lower()
    
    logging.info(f"Processing structured file: {file_name}")
    conn = get_db_connection(db_url)
    
    try:
        # 1. Read File
        sheets_map = {}
        if ext == 'csv':
            try:
                df = pd.read_csv(file_path)
                sheets_map['main'] = df
            except Exception as e:
                logging.error(f"Failed to read CSV: {e}")
                return
        else: # xlsx, xls
            try:
                xl = pd.ExcelFile(file_path)
                for s_name in xl.sheet_names:
                    sheets_map[s_name] = xl.parse(s_name)
            except Exception as e:
                logging.error(f"Failed to read Excel: {e}")
                return

        # 2. Prepare Metadata & Embeddings
        # We process all sheets to build batch lists
        
        # File Metadata Heuristic
        file_summary = f"Financial file '{file_name}' containing {len(sheets_map)} sheets: {', '.join(sheets_map.keys())}."
        file_keywords = f"finance, {ext}, {file_name}"
        
        payload_queue = [file_summary, file_keywords]
        
        tables_to_create = []
        
        for sheet_name, df in sheets_map.items():
            if df.empty: continue
            
            # Normalize Columns
            df.columns = [normalize_name(c) for c in df.columns]
            # De-duplicate
            seen = {}
            new_cols = []
            for c in df.columns:
                if c in seen:
                    seen[c]+=1
                    c = f"{c}_{seen[c]}"
                else:
                    seen[c]=0
                new_cols.append(c)
            df.columns = new_cols
            
            # Heuristics
            summ, keys = generate_heuristics(df, sheet_name)
            
            # Column string for embedding
            col_meta_str = f"Columns in {sheet_name}: " + ", ".join([f"{c} ({df[c].dtype})" for c in df.columns])
            
            payload_queue.extend([summ, keys, col_meta_str])
            
            tables_to_create.append({
                "sheet_name": sheet_name,
                "df": df,
                "summary": summ,
                "keywords": keys,
                "col_meta": col_meta_str
            })
            
        # 3. Batch Embeddings
        logging.info("Generating embeddings...")
        embeddings = get_embeddings_batch(payload_queue, openai_key)
        
        f_summ_emb = embeddings[0]
        f_key_emb = embeddings[1]
        
        # 4. DB Transactions
        with conn.cursor() as cur:
            # Ensure Schema
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS files_metadata (
                    file_id UUID PRIMARY KEY,
                    file_name TEXT,
                    file_type TEXT,
                    summary TEXT,
                    keywords TEXT,
                    summary_embedding vector(3072),
                    keywords_embedding vector(3072),
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sheets_metadata (
                    sheet_id UUID PRIMARY KEY,
                    file_id UUID REFERENCES files_metadata(file_id) ON DELETE CASCADE,
                    sheet_name TEXT,
                    table_name TEXT,
                    num_rows INT,
                    summary TEXT,
                    keywords TEXT,
                    summary_embedding vector(3072),
                    keywords_embedding vector(3072),
                    columns_embedding vector(3072)
                );
            """)
            
            # Create File Record
            cur.execute("""
                INSERT INTO files_metadata (file_id, file_name, file_type, summary, keywords, summary_embedding, keywords_embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (str(file_id), file_name, ext, file_summary, file_keywords, f_summ_emb, f_key_emb))
            
            current_emb_idx = 2
            
            for tbl in tables_to_create:
                df = tbl['df']
                
                # Get embeddings from result list
                s_emb = embeddings[current_emb_idx]
                k_emb = embeddings[current_emb_idx+1]
                c_emb = embeddings[current_emb_idx+2]
                current_emb_idx += 3
                
                # Dynamic Table Creation
                short_id = str(file_id).replace('-','')[:8]
                s_safe = normalize_name(tbl['sheet_name'])
                table_name = f"data_{short_id}_{s_safe}"
                
                # Create Table SQL
                col_defs = []
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    if 'int' in dtype: pgtype = 'BIGINT'
                    elif 'float' in dtype: pgtype = 'DOUBLE PRECISION'
                    else: pgtype = 'TEXT'
                    col_defs.append(f'"{col}" {pgtype}')
                    
                create_sql = f"CREATE TABLE {table_name} ({', '.join(col_defs)});"
                cur.execute(create_sql)
                
                # Insert Data
                # Replace nan with None
                df_clean = df.where(pd.notnull(df), None)
                vals = [tuple(x) for x in df_clean.to_numpy()]
                cols_sql = ", ".join([f'"{c}"' for c in df.columns])
                placeholders = ", ".join(["%s"] * len(df.columns))
                insert_sql = f"INSERT INTO {table_name} ({cols_sql}) VALUES ({placeholders})"
                
                # Batch insert (executemany)
                cur.executemany(insert_sql, vals)
                
                # Sheet Metadata
                cur.execute("""
                    INSERT INTO sheets_metadata (sheet_id, file_id, sheet_name, table_name, num_rows, summary, keywords, summary_embedding, keywords_embedding, columns_embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()), str(file_id), tbl['sheet_name'], table_name, len(df),
                    tbl['summary'], tbl['keywords'], s_emb, k_emb, c_emb
                ))
            
            conn.commit()
            logging.info(f"Ingestion complete for {file_name}")

    except Exception as e:
        conn.rollback()
        logging.error(f"Error processing structured file: {e}")
        raise e
    finally:
        conn.close()
