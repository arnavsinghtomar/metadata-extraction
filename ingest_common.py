
import os
import psycopg2
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load env immediately
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_db_connection(db_url):
    """
    Establish reliable connection to Postgres.
    """
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        raise e

def get_embedding(text, api_key):
    """
    Get single embedding vector.
    """
    if not text:
        return None
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        response = client.embeddings.create(
            input=text,
            model="openai/text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Embedding failed: {e}")
        return None

def get_embeddings_batch(texts, api_key):
    """
    Get multiple embeddings in one batch.
    """
    if not texts:
        return []
        
    valid_texts = [t for t in texts if t]
    if not valid_texts:
        return [None] * len(texts) # Preserve order

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        # Limit batch size if needed, but usually fine for <100
        response = client.embeddings.create(
            input=valid_texts,
            model="openai/text-embedding-3-small"
        )
        
        # Map results back to original list (handling empty strings)
        embedding_map = {i: data.embedding for i, data in enumerate(response.data)}
        
        final_results = []
        valid_idx = 0
        for t in texts:
            if t:
                final_results.append(embedding_map[valid_idx])
                valid_idx += 1
            else:
                final_results.append(None)
        return final_results

    except Exception as e:
        logging.error(f"Batch embedding failed: {e}")
        return [None] * len(texts)
