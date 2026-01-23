import psycopg2, os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
print("✅ Connected to Neon PostgreSQL")
conn.close()