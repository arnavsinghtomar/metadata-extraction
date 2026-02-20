"""
Unified File Ingestion Router

This module provides a central entry point for ingesting various file types.
It detects the file type and routes to the appropriate specialized processor.

Supported formats:
- Excel (.xlsx, .xls)
- CSV (.csv)
- PDF (.pdf)
- Text (.txt)
- Word (.doc, .docx)
"""

import os
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# File type constants
FILE_TYPE_EXCEL = 'excel'
FILE_TYPE_CSV = 'csv'
FILE_TYPE_PDF = 'pdf'
FILE_TYPE_TEXT = 'text'
FILE_TYPE_WORD = 'word'
FILE_TYPE_UNKNOWN = 'unknown'

# Extension mapping
EXTENSION_MAP = {
    '.xlsx': FILE_TYPE_EXCEL,
    '.xls': FILE_TYPE_EXCEL,
    '.csv': FILE_TYPE_CSV,
    '.pdf': FILE_TYPE_PDF,
    '.txt': FILE_TYPE_TEXT,
    '.doc': FILE_TYPE_WORD,
    '.docx': FILE_TYPE_WORD,
}

def detect_file_type(file_path: str) -> str:
    """
    Detect file type based on extension.
    
    Args:
        file_path: Path to file
    
    Returns:
        File type constant (FILE_TYPE_*)
    """
    ext = os.path.splitext(file_path)[1].lower()
    return EXTENSION_MAP.get(ext, FILE_TYPE_UNKNOWN)

def process_file(
    file_path: str,
    db_url: str,
    openai_key: str,
    data_domain: str = "general",
    sensitivity_level: str = "internal",
    original_filename: Optional[str] = None,
    skip_ai_analysis: bool = False
) -> Dict:
    """
    Unified file ingestion entry point.
    
    Detects file type and routes to appropriate processor.
    
    Args:
        file_path: Path to file to ingest
        db_url: Database connection URL
        openai_key: OpenAI/OpenRouter API key
        data_domain: Data domain for RBAC (finance, hr, sales, etc.)
        sensitivity_level: Sensitivity level (public, internal, confidential, restricted)
        original_filename: Original filename (for uploaded files)
        skip_ai_analysis: Skip AI analysis for faster processing (Excel only)
    
    Returns:
        Dict with status and metadata:
        {
            'status': 'SUCCESS' | 'DUPLICATE' | 'EMPTY' | 'ERROR',
            'file_type': str,
            'message': str,
            'file_id': str (if successful)
        }
    """
    file_type = detect_file_type(file_path)
    filename = original_filename or os.path.basename(file_path)
    
    logging.info(f"Processing file: {filename} (Type: {file_type})")
    
    try:
        if file_type == FILE_TYPE_EXCEL:
            from ingest_excel import process_excel_file
            
            status = process_excel_file(
                file_path=file_path,
                db_url=db_url,
                openai_key=openai_key,
                data_domain=data_domain,
                sensitivity_level=sensitivity_level,
                original_filename=original_filename,
                skip_ai_analysis=skip_ai_analysis
            )
            
            return {
                'status': status,
                'file_type': file_type,
                'message': f"Excel file processed: {filename}"
            }
        
        elif file_type == FILE_TYPE_CSV:
            from ingest_structured import process_structured_file
            
            process_structured_file(
                file_path=file_path,
                db_url=db_url,
                openai_key=openai_key,
                original_filename=original_filename
            )
            
            return {
                'status': 'SUCCESS',
                'file_type': file_type,
                'message': f"CSV file processed: {filename}"
            }
        
        elif file_type == FILE_TYPE_PDF:
            from ingest_pdf import process_pdf_file, create_pdf_tables
            
            # Ensure PDF tables exist
            from ingest_common import get_db_connection
            conn = get_db_connection(db_url)
            create_pdf_tables(conn)
            conn.close()
            
            process_pdf_file(
                file_path=file_path,
                db_url=db_url,
                openai_key=openai_key
            )
            
            return {
                'status': 'SUCCESS',
                'file_type': file_type,
                'message': f"PDF file processed: {filename}"
            }
        
        elif file_type == FILE_TYPE_TEXT:
            from ingest_text import process_text_file, create_text_tables
            
            # Ensure text tables exist
            from ingest_common import get_db_connection
            conn = get_db_connection(db_url)
            create_text_tables(conn)
            conn.close()
            
            status = process_text_file(
                file_path=file_path,
                db_url=db_url,
                openai_key=openai_key,
                data_domain=data_domain,
                sensitivity_level=sensitivity_level,
                original_filename=original_filename
            )
            
            return {
                'status': status,
                'file_type': file_type,
                'message': f"Text file processed: {filename}"
            }
        
        elif file_type == FILE_TYPE_WORD:
            from ingest_word import process_word_file, create_word_tables
            
            # Ensure Word tables exist
            from ingest_common import get_db_connection
            conn = get_db_connection(db_url)
            create_word_tables(conn)
            conn.close()
            
            status = process_word_file(
                file_path=file_path,
                db_url=db_url,
                openai_key=openai_key,
                data_domain=data_domain,
                sensitivity_level=sensitivity_level,
                original_filename=original_filename
            )
            
            return {
                'status': status,
                'file_type': file_type,
                'message': f"Word document processed: {filename}"
            }
        
        else:
            error_msg = f"Unsupported file type: {filename}"
            logging.error(error_msg)
            return {
                'status': 'ERROR',
                'file_type': FILE_TYPE_UNKNOWN,
                'message': error_msg
            }
    
    except Exception as e:
        error_msg = f"Error processing {filename}: {str(e)}"
        logging.error(error_msg)
        return {
            'status': 'ERROR',
            'file_type': file_type,
            'message': error_msg,
            'error': str(e)
        }

def get_supported_extensions():
    """
    Get list of supported file extensions.
    
    Returns:
        List of extensions (with dots)
    """
    return list(EXTENSION_MAP.keys())

def get_supported_types():
    """
    Get list of supported file types.
    
    Returns:
        List of file type constants
    """
    return list(set(EXTENSION_MAP.values()))

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    load_dotenv()
    
    if len(sys.argv) != 2:
        print("Usage: python ingest_unified.py <path_to_file>")
        print(f"Supported extensions: {', '.join(get_supported_extensions())}")
        sys.exit(1)
    
    file_path = sys.argv[1]
    db_url = os.getenv("DATABASE_URL")
    openai_key = os.getenv("OPENROUTER_API_KEY")
    
    if not db_url:
        logging.error("DATABASE_URL not found in environment")
        sys.exit(1)
    
    result = process_file(file_path, db_url, openai_key)
    
    print(f"\nResult: {result['status']}")
    print(f"Message: {result['message']}")
    
    if result['status'] == 'SUCCESS':
        print(f"✅ File successfully ingested!")
    else:
        print(f"❌ Ingestion failed")
        sys.exit(1)
