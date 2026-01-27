"""
Script to fix vector dimension mismatch by cleaning up the database.
This will delete all existing data and recreate tables with correct dimensions.
"""

import os
import sys
import logging
from dotenv import load_dotenv
from cleanup import cleanup_database
from ingest_excel import get_db_connection, create_metadata_tables

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fix_vector_dimensions():
    """
    Fix vector dimension mismatch by:
    1. Cleaning up all existing data
    2. Recreating tables with correct 3072 dimensions
    """
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    
    if not db_url:
        logging.error("DATABASE_URL not found in .env file")
        sys.exit(1)
    
    print("=" * 60)
    print("VECTOR DIMENSION FIX UTILITY")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Delete ALL existing data from the database")
    print("2. Recreate tables with correct 3072-dimensional vectors")
    print("\nWARNING: This is IRREVERSIBLE!")
    print("=" * 60)
    
    confirm = input("\nType 'yes' to proceed: ").lower().strip()
    
    if confirm != 'yes':
        print("Operation cancelled.")
        sys.exit(0)
    
    try:
        # Step 1: Clean up existing data
        logging.info("Step 1: Cleaning up existing data...")
        cleanup_database(db_url)
        
        # Step 2: Recreate tables with correct dimensions
        logging.info("Step 2: Recreating tables with 3072-dimensional vectors...")
        conn = get_db_connection(db_url)
        create_metadata_tables(conn)
        conn.close()
        
        logging.info("✅ Vector dimensions fixed successfully!")
        logging.info("You can now upload files without dimension mismatch errors.")
        
    except Exception as e:
        logging.error(f"❌ Error during fix: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fix_vector_dimensions()
