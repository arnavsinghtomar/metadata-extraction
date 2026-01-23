import os
import sys
import logging
import psycopg2
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_environment():
    """Load environment variables from .env file."""
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logging.error("DATABASE_URL not found in .env environment.")
        sys.exit(1)
    return db_url

def get_db_connection(db_url):
    """Connect to the Postgres database."""
    try:
        conn = psycopg2.connect(db_url)
        return conn
    except Exception as e:
        logging.error(f"Failed to connect to database: {e}")
        sys.exit(1)

def cleanup_database(db_url):
    """
    Remove all dynamic sheet tables and clear metadata.
    """
    conn = get_db_connection(db_url)
    
    try:
        cur = conn.cursor()
        
        # 1. Get all dynamic table names from metadata
        cur.execute("SELECT to_regclass('sheets_metadata');")
        if not cur.fetchone()[0]:
            logging.info("Metadata tables do not exist. Skipping metadata check.")
        else:
            # Drop all tables that start with 'sheet_' (our data tables)
            # Fetch from information_schema to be thorough, not just what's in metadata
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name LIKE 'sheet_%'
            """)
            tables = [row[0] for row in cur.fetchall()]
            
            if not tables:
                logging.info("No 'sheet_%' tables found in schema.")
            else:
                logging.info(f"Found {len(tables)} data tables to drop.")
                # PostgreSQL can handle many tables in one DROP statement.
                # Splitting into chunks of 1000 to be safe but efficient.
                batch_size = 1000
                for i in range(0, len(tables), batch_size):
                    batch = tables[i:i + batch_size]
                    tables_str = ", ".join([f'"{t}"' for t in batch]) # Quote identifiers
                    logging.info(f"Dropping batch {i//batch_size + 1} ({len(batch)} tables)...")
                    cur.execute(f"DROP TABLE IF EXISTS {tables_str} CASCADE")
        
        # 3. Clear metadata tables
        # Using TRUNCATE ... CASCADE to ensure reference integrity is handled (clears sheets_metadata too)
        logging.info("Clearing metadata tables (files_metadata, sheets_metadata)...")
        cur.execute("TRUNCATE TABLE files_metadata CASCADE")
        
        conn.commit()
        logging.info("Database cleanup completed successfully.")
        
    except Exception as e:
        conn.rollback()
        logging.error(f"An error occurred during cleanup: {e}")
    finally:
        conn.close()
        logging.info("Database connection closed.")

if __name__ == "__main__":
    force_run = "--force" in sys.argv
    
    if force_run:
        print("Force flag detected. Proceeding with cleanup...")
        db_url_env = load_environment()
        cleanup_database(db_url_env)
    else:
        print("WARNING: This will DELETE ALL INGESTED DATA and METADATA.")
        confirm = input("Are you sure you want to proceed? (yes/no): ").lower()
        
        if confirm == 'yes':
            db_url_env = load_environment()
            cleanup_database(db_url_env)
        else:
            print("Cleanup aborted.")