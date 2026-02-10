"""
FastAPI dependencies for dependency injection
"""

import os
from typing import Generator
from dotenv import load_dotenv
from fastapi import HTTPException
import psycopg2
from agents.ingestion_agent import IngestionAgent

# Load environment variables
load_dotenv()

# Environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_KEY = os.getenv("OPENROUTER_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not found in environment variables")

if not OPENAI_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not found in environment variables")


def get_database_url() -> str:
    """Get database URL from environment"""
    return DATABASE_URL


def get_openai_key() -> str:
    """Get OpenAI API key from environment"""
    return OPENAI_KEY


def get_db_connection() -> Generator:
    """
    Dependency for database connection
    Yields a connection and ensures it's closed after use
    """
    from ingest_excel import get_db_connection as get_conn
    
    conn = None
    try:
        conn = get_conn(DATABASE_URL)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database connection error: {str(e)}")
    finally:
        if conn:
            conn.close()


def get_ingestion_agent() -> IngestionAgent:
    """
    Dependency for IngestionAgent
    Returns a configured agent instance
    """
    try:
        agent = IngestionAgent(
            db_url=DATABASE_URL,
            openai_key=OPENAI_KEY
        )
        return agent
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize IngestionAgent: {str(e)}")


def validate_file_extension(filename: str, allowed_extensions: set = {'.xlsx', '.xls', '.pdf', '.csv'}) -> bool:
    """
    Validate file extension
    """
    import os
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_extensions


def get_file_extension(filename: str) -> str:
    """
    Get file extension
    """
    import os
    return os.path.splitext(filename)[1].lower()
