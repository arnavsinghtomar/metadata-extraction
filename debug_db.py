import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

def debug():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    print("--- Retrieval Policies ---")
    cur.execute("SELECT role, data_domain, sensitivity_level, allowed FROM retrieval_policies")
    for row in cur.fetchall():
        print(row)
        
    print("\n--- Sheets Metadata ---")
    cur.execute("SELECT sheet_name, data_domain, sensitivity_level FROM sheets_metadata")
    for row in cur.fetchall():
        print(row)
        
    conn.close()

if __name__ == "__main__":
    debug()
