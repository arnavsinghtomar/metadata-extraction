import streamlit as st
import pandas as pd
import psycopg2
import os
import tempfile
from dotenv import load_dotenv
from ingest_excel import process_excel_file, create_metadata_tables, get_db_connection
from cleanup import cleanup_database

# Page Config
st.set_page_config(
    page_title="Financial Data Extraction",
    page_icon="📊",
    layout="wide"
)

# Load Environment
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

if not DB_URL:
    st.error("DATABASE_URL not found in .env file.")
    st.stop()

# Helper to get connection
def get_conn():
    return get_db_connection(DB_URL)

# Title
st.title("📊 Financial Data Manager")

# Sidebar - Actions
st.sidebar.header("Actions")

# 1. Cleanup Section
st.sidebar.subheader("Danger Zone")
if st.sidebar.button("🗑️ Delete All Data", type="primary"):
    with st.spinner("Cleaning database..."):
        try:
            cleanup_database(DB_URL)
            st.sidebar.success("Database cleared!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error cleaning database: {e}")

# 2. Upload Section
st.sidebar.subheader("Ingest Data")
uploaded_files = st.sidebar.file_uploader("Upload Excel Files (Max 3)", type=["xlsx", "xls"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 3:
        st.sidebar.error("Maximum 3 files allowed. Please remove some.")
    else:
        if st.sidebar.button("Process Files"):
            # Ensure metadata tables exist (idempotent)
            conn = get_conn()
            create_metadata_tables(conn)
            conn.close()

            progress_bar = st.sidebar.progress(0)
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.sidebar.write(f"Processing {uploaded_file.name}...")
                try:
                    # Save to temp file because our ingest script expects a path
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Run ingestion
                    process_excel_file(tmp_path, DB_URL, OPENROUTER_KEY)
                    
                    st.sidebar.success(f"Successfully processed {uploaded_file.name}")
                    os.remove(tmp_path)
                    
                except Exception as e:
                    st.sidebar.error(f"Error processing {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.rerun()

# Main Content - Data Explorer
st.header("Database Content")

conn = get_conn()

# Ensure schema is up to date (runs migrations if needed)
try:
    create_metadata_tables(conn)
except Exception as e:
    st.error(f"Schema migration failed: {e}")

# Check if metadata tables exist
table_check_query = "SELECT to_regclass('files_metadata');"
cursor = conn.cursor()
cursor.execute(table_check_query)
exists = cursor.fetchone()[0]

if not exists:
    st.warning("Database is empty. Please upload a file to get started.")
else:
    # 1. Files
    st.subheader("📁 Ingested Files")
    files_query = """
    SELECT 
        file_id, 
        file_name, 
        uploaded_at, 
        num_sheets,
        summary,
        keywords
    FROM files_metadata 
    ORDER BY uploaded_at DESC
    """
    files_df = pd.read_sql(files_query, conn)
    
    if files_df.empty:
        st.info("No files found.")
    else:
        # Custom display for files with metadata
        for _, row in files_df.iterrows():
            with st.expander(f"📁 {row['file_name']} ({row['uploaded_at'].strftime('%Y-%m-%d %H:%M')})", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Sheets:** {row['num_sheets']}")
                    st.markdown(f"**Keywords:** {row['keywords']}")
                with col2:
                    st.markdown(f"**Summary:** {row['summary']}")
        

# 2. Sheets Explorer
st.subheader("📑 Sheets Explorer")

# Join sheets with files to get file name
sheets_query = """
SELECT 
    s.sheet_id,
    f.file_name, 
    s.sheet_name, 
    s.table_name, 
    s.num_rows, 
    s.num_columns,
    s.summary,
    s.category,
    s.keywords
FROM sheets_metadata s
JOIN files_metadata f ON s.file_id = f.file_id
ORDER BY f.uploaded_at DESC, s.sheet_name
"""
sheets_df = pd.read_sql(sheets_query, conn)

if not sheets_df.empty:
    with st.expander("📑 Raw Data Explorer", expanded=False):
        # Selection for drill down
        selected_sheet = st.selectbox(
            "Select a sheet to view data:", 
            options=sheets_df['table_name'].tolist(),
            format_func=lambda x: f"[{sheets_df[sheets_df['table_name'] == x].iloc[0]['category'] or 'Uncategorized'}] {sheets_df[sheets_df['table_name'] == x].iloc[0]['file_name']} - {sheets_df[sheets_df['table_name'] == x].iloc[0]['sheet_name']}"
        )
        
        # Show Metadata for selected
        sel_meta = sheets_df[sheets_df['table_name'] == selected_sheet].iloc[0]
        st.info(f"**Category:** {sel_meta['category']} | **Keywords:** {sel_meta['keywords']}")
        
        # Fetch data from the dynamic table
        limit = st.slider("Rows to fetch", min_value=10, max_value=1000, value=50)
        data_query = f"SELECT * FROM {selected_sheet} LIMIT {limit}"
        try:
            sheet_data = pd.read_sql(data_query, conn)
            st.dataframe(sheet_data, width="stretch")
        except Exception as e:
            st.error(f"Could not read table: {e}")

    # ==========================================
    # 3. Chat Interface (Retrieval Strategy)
    # ==========================================
    st.divider()
    st.subheader("🤖 Ask Your Data (Beta)")
    
    with st.form("chat_form"):
        user_query = st.text_input("Ask a question about your files (e.g., 'What was the total revenue in 2024?')")
        submitted = st.form_submit_button("Ask Agent")
        
    if submitted and user_query:
        from retrieval import process_retrieval
        
        with st.spinner("🤖 Analyzing schema and generating query..."):
            try:
                # Run the retrieval pipeline
                result_pack = process_retrieval(user_query, DB_URL, OPENROUTER_KEY)
                
                if "error" in result_pack:
                    st.error(result_pack["error"])
                else:
                    # 1. Show the Answer
                    st.success(f"**Answer:** {result_pack['final_answer']}")
                    
                    # 1b. Show Plotly Chart if available
                    if "chart" in result_pack:
                        st.plotly_chart(result_pack["chart"], width="stretch")
                    
                    # 2. Show the "Work" (Expander)
                    with st.expander("🕵️ View Agent's Thought Process"):
                        
                        # Step 1: Sheet Selection
                        sheet = result_pack['sheet_match']
                        st.markdown(f"**1. Selected Source:** `{sheet['sheet_name']}` (Similarity Score: {sheet['distance']:.4f})")
                        st.caption(f"Table: {sheet['table_name']}")
                        
                        # Step 2: SQL Generation
                        st.markdown("**2. Generated SQL:**")
                        st.code(result_pack['generated_sql'], language="sql")
                        
                        # Step 3: Raw Results
                        st.markdown("**3. Raw Data Results:**")
                        st.dataframe(result_pack['results_df'], width="stretch")
                        
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

else:
    st.info("No sheets found to analyze.")

conn.close()
