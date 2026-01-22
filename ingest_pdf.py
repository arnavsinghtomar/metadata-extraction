
import os
import sys
import uuid
import logging
import json
from pypdf import PdfReader
import psycopg2
from psycopg2 import sql
from openai import OpenAI

# Import shared utilities from ingest_excel
# We assume ingest_excel is in the same directory and accessible
try:
    from ingest_excel import get_db_connection, get_embeddings_batch, load_environment
except ImportError:
    # Log error if we can't import, but we shouldn't fail silently
    logging.error("Could not import logic from ingest_excel.py. Make sure it is in the path.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_pdf_tables(conn):
    """
    Create the PDF-specific metadata tables.
    We rely on 'files_metadata' existing (created by ingest_excel logic), 
    but we can add creation here too just in case.
    """
    with conn.cursor() as cur:
        # Ensure vector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Ensure files_metadata exists (it might be shared)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS files_metadata (
                file_id UUID PRIMARY KEY,
                file_name TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                num_sheets INTEGER,
                summary TEXT,
                keywords TEXT,
                summary_embedding vector(1536),
                keywords_embedding vector(1536)
            );
        """)
        
        # Add num_pages column if it doesn't exist (for PDF support)
        cur.execute("""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='files_metadata' AND column_name='num_pages') THEN 
                    ALTER TABLE files_metadata ADD COLUMN num_pages INTEGER; 
                END IF; 
            END $$;
        """)
        
        # PDF Chunks Table for RAG
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pdf_chunks (
                chunk_id UUID PRIMARY KEY,
                file_id UUID REFERENCES files_metadata(file_id) ON DELETE CASCADE,
                page_number INTEGER,
                chunk_index INTEGER,
                chunk_text TEXT,
                embedding vector(1536)
            );
        """)
        
        # We might want an HNSW index for speed later, but basic is fine for now
        
    conn.commit()
    logging.info("PDF metadata tables verified.")

def extract_text_from_pdf(file_path):
    """
    Extract text from PDF pages.
    Returns: list of (page_num, text) tuples.
    """
    pages_content = []
    try:
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_content.append((i + 1, text.strip()))
    except Exception as e:
        logging.error(f"Error reading PDF {file_path}: {e}")
        raise e
        
    return pages_content

def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Simple character-based chunking.
    Could be improved with semantic splitters later.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
        if start >= len(text):
            break
            
    return chunks

def process_pdf_file(file_path, db_url, openai_key):
    """
    Main logic to ingest a PDF file.
    1. Parse PDF
    2. Chunk content
    3. Generate Embeddings (Batch)
    4. Store in DB
    """
    file_path = os.path.abspath(file_path)
    file_name = os.path.basename(file_path)
    file_id = uuid.uuid4()
    
    logging.info(f"Processing PDF file: {file_name} (ID: {file_id})")
    conn = get_db_connection(db_url)
    
    try:
        # 1. Extract Text
        pages = extract_text_from_pdf(file_path)
        if not pages:
            logging.warning(f"No text extracted from {file_name}. It might be empty or scanned images.")
            return

        # 2. Prepare Chunks & Metadata
        # We will create one giant list of text to embed:
        # [File Summary, File Keywords, Chunk1, Chunk2, ...]
        
        # Generate File Summary (Heuristic: First 2000 chars of text)
        full_text_sample = " ".join([p[1] for p in pages[:3]])[:2000]
        file_summary = f"PDF Document '{file_name}' containing {len(pages)} pages. Preview: {full_text_sample[:200]}..."
        file_keywords = f"PDF, {file_name}, Document"
        
        all_chunks = []
        embedding_payloads = []
        
        # Add metadata first
        embedding_payloads.append(file_summary)
        embedding_payloads.append(file_keywords)
        
        chunk_metadata_map = [] # To map index back to page/chunk info
        
        for page_num, page_text in pages:
            page_chunks = chunk_text(page_text)
            for idx, c_text in enumerate(page_chunks):
                clean_text = c_text.replace('\0', '') # Postgres doesn't like null bytes
                all_chunks.append({
                    "page": page_num,
                    "chunk_index": idx,
                    "text": clean_text
                })
                embedding_payloads.append(clean_text)
                
        # 3. Batch Embeddings
        logging.info(f"Generating embeddings for {len(embedding_payloads)} items...")
        embeddings = get_embeddings_batch(embedding_payloads, openai_key)
        
        f_summ_emb = embeddings[0]
        f_key_emb = embeddings[1]
        chunk_embeddings = embeddings[2:]
        
        # 4. Database Transaction
        with conn.cursor() as cur:
            # A. Insert File Metadata
            cur.execute(
                """
                INSERT INTO files_metadata (file_id, file_name, num_pages, summary, keywords, summary_embedding, keywords_embedding) 
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (str(file_id), file_name, len(pages), file_summary, file_keywords, f_summ_emb, f_key_emb)
            )
            
            # B. Insert Chunks
            insert_query = """
                INSERT INTO pdf_chunks (chunk_id, file_id, page_number, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            data_to_insert = []
            for i, chunk in enumerate(all_chunks):
                c_id = str(uuid.uuid4())
                c_emb = chunk_embeddings[i]
                data_to_insert.append((c_id, str(file_id), chunk['page'], chunk['chunk_index'], chunk['text'], c_emb))
                
            cur.executemany(insert_query, data_to_insert)
            
        conn.commit()
        logging.info(f"Successfully ingested {file_name} with {len(all_chunks)} chunks.")
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error processing PDF {file_name}: {e}")
        # Re-raise to alert caller/UI
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_pdf.py <path_to_pdf>")
        sys.exit(1)
        
    pdf_file = sys.argv[1]
    db_url_env, openai_key_env = load_environment()
    
    # Init tables
    con = get_db_connection(db_url_env)
    create_pdf_tables(con)
    con.close()
    
    process_pdf_file(pdf_file, db_url_env, openai_key_env)
