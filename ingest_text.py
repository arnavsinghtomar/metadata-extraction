
import os
import sys
import uuid
import re
import logging
import json
import chardet
import psycopg2
from openai import OpenAI

# Import shared utilities
try:
    from ingest_common import get_db_connection, get_embeddings_batch
except ImportError:
    logging.error("Could not import from ingest_common.py")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def detect_encoding(file_path):
    """
    Detect file encoding using chardet.
    """
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read first 10KB
        result = chardet.detect(raw_data)
        return result['encoding'] or 'utf-8'

def extract_financial_entities(text):
    """
    Extract financial entities from text using regex patterns.
    Returns dict with extracted entities.
    """
    entities = {
        'amounts': [],
        'dates': [],
        'percentages': [],
        'companies': []
    }
    
    # Extract monetary amounts (e.g., $1,234.56, USD 1000, Rs. 5000)
    amount_patterns = [
        r'\$\s*[\d,]+\.?\d*',
        r'USD\s*[\d,]+\.?\d*',
        r'Rs\.?\s*[\d,]+\.?\d*',
        r'INR\s*[\d,]+\.?\d*',
        r'€\s*[\d,]+\.?\d*',
        r'£\s*[\d,]+\.?\d*'
    ]
    for pattern in amount_patterns:
        entities['amounts'].extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Extract dates (various formats)
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
        r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}'
    ]
    for pattern in date_patterns:
        entities['dates'].extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Extract percentages
    entities['percentages'] = re.findall(r'\d+\.?\d*\s*%', text)
    
    # Extract potential company names (capitalized words, 2-4 words)
    # This is a simple heuristic
    company_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:Inc|Corp|Ltd|LLC|Limited|Corporation|Company)'
    entities['companies'] = re.findall(company_pattern, text)
    
    return entities

def chunk_text_semantic(text, chunk_size=1000, overlap=200):
    """
    Chunk text by paragraphs and sections, respecting semantic boundaries.
    """
    chunks = []
    
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n', text)
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding this paragraph exceeds chunk size, save current chunk
        if len(current_chunk) + len(para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            words = current_chunk.split()
            overlap_text = ' '.join(words[-overlap:]) if len(words) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def generate_text_summary(text, file_name, openai_key):
    """
    Generate AI summary and keywords for text document.
    """
    if not openai_key:
        return None
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openai_key
        )
        
        # Use first 2000 chars for analysis
        sample = text[:2000]
        
        prompt = (
            f"Analyze this financial text document named '{file_name}'.\n\n"
            f"Content preview:\n{sample}\n\n"
            f"Return a JSON object with:\n"
            f"1. 'category': Document category (e.g., 'Financial Report', 'Invoice', 'Statement', 'Memo')\n"
            f"2. 'summary': A concise 2-3 sentence summary\n"
            f"3. 'keywords': 5-10 keywords as comma-separated string"
        )
        
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial document analyst. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logging.error(f"Failed to generate text summary: {e}")
        return None

def create_text_tables(conn):
    """
    Create text document tables in database.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Ensure files_metadata exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS files_metadata (
                file_id UUID PRIMARY KEY,
                file_name TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                num_sheets INTEGER,
                num_pages INTEGER,
                summary TEXT,
                keywords TEXT,
                summary_embedding vector(3072),
                keywords_embedding vector(3072)
            );
        """)
        
        # Text documents chunks table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS text_documents (
                doc_id UUID PRIMARY KEY,
                file_id UUID REFERENCES files_metadata(file_id) ON DELETE CASCADE,
                chunk_index INTEGER,
                chunk_text TEXT,
                entities JSONB,
                embedding vector(3072)
            );
        """)
    
    conn.commit()
    logging.info("Text document tables verified.")

def process_text_file(file_path, db_url, openai_key, data_domain="general", sensitivity_level="internal", original_filename=None):
    """
    Main logic to ingest text file.
    
    Args:
        file_path: Path to text file
        db_url: Database URL
        openai_key: OpenAI API key
        data_domain: Data domain for RBAC
        sensitivity_level: Sensitivity level for RBAC
        original_filename: Original filename (for uploads)
    
    Returns: "SUCCESS" or raises Exception
    """
    file_path = os.path.abspath(file_path)
    file_name = original_filename if original_filename else os.path.basename(file_path)
    file_id = uuid.uuid4()
    
    logging.info(f"Processing text file: {file_name} (ID: {file_id})")
    
    conn = get_db_connection(db_url)
    
    try:
        # 1. Detect encoding and read file
        encoding = detect_encoding(file_path)
        logging.info(f"Detected encoding: {encoding}")
        
        with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
            full_text = f.read()
        
        if not full_text.strip():
            logging.warning(f"File {file_name} is empty")
            return "EMPTY"
        
        # 2. Generate file-level metadata
        ai_analysis = generate_text_summary(full_text, file_name, openai_key)
        
        if ai_analysis:
            category = ai_analysis.get('category', 'Text Document')
            file_summary = ai_analysis.get('summary', f"Text document: {file_name}")
            file_keywords = ai_analysis.get('keywords', file_name)
        else:
            category = 'Text Document'
            file_summary = f"Text document: {file_name}"
            file_keywords = file_name
        
        # 3. Chunk text
        chunks = chunk_text_semantic(full_text)
        logging.info(f"Created {len(chunks)} chunks")
        
        # 4. Extract entities from each chunk
        chunk_data = []
        for idx, chunk_text in enumerate(chunks):
            entities = extract_financial_entities(chunk_text)
            chunk_data.append({
                'index': idx,
                'text': chunk_text,
                'entities': entities
            })
        
        # 5. Prepare embeddings batch
        embedding_texts = [file_summary, file_keywords]
        for chunk in chunk_data:
            embedding_texts.append(chunk['text'])
        
        # 6. Generate embeddings
        logging.info(f"Generating embeddings for {len(embedding_texts)} items...")
        embeddings = get_embeddings_batch(embedding_texts, openai_key)
        
        file_summary_emb = embeddings[0]
        file_keywords_emb = embeddings[1]
        chunk_embeddings = embeddings[2:]
        
        # 7. Database transaction
        with conn.cursor() as cur:
            # Insert file metadata
            cur.execute("""
                INSERT INTO files_metadata 
                (file_id, file_name, summary, keywords, summary_embedding, keywords_embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (str(file_id), file_name, file_summary, file_keywords, file_summary_emb, file_keywords_emb))
            
            # Insert chunks
            insert_query = """
                INSERT INTO text_documents 
                (doc_id, file_id, chunk_index, chunk_text, entities, embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            chunk_records = []
            for i, chunk in enumerate(chunk_data):
                doc_id = str(uuid.uuid4())
                chunk_records.append((
                    doc_id,
                    str(file_id),
                    chunk['index'],
                    chunk['text'],
                    json.dumps(chunk['entities']),
                    chunk_embeddings[i]
                ))
            
            cur.executemany(insert_query, chunk_records)
        
        conn.commit()
        logging.info(f"Successfully ingested {file_name} with {len(chunks)} chunks")
        return "SUCCESS"
        
    except Exception as e:
        conn.rollback()
        logging.error(f"Error processing text file {file_name}: {e}")
        raise e
    finally:
        conn.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_text.py <path_to_text_file>")
        sys.exit(1)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    text_file = sys.argv[1]
    db_url = os.getenv("DATABASE_URL")
    openai_key = os.getenv("OPENROUTER_API_KEY")
    
    if not db_url:
        logging.error("DATABASE_URL not found in environment")
        sys.exit(1)
    
    # Init tables
    conn = get_db_connection(db_url)
    create_text_tables(conn)
    conn.close()
    
    process_text_file(text_file, db_url, openai_key)
