import io
import streamlit as st
import pandas as pd
import psycopg2
import os
import tempfile
import concurrent.futures
import time
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from ingest_excel import process_excel_file, create_metadata_tables, get_db_connection
from cleanup import cleanup_database
from analytics import compute_business_health

# Page Config
st.set_page_config(
    page_title="Financial Data Extraction",
    page_icon="📊",
    layout="wide"
)

# Google Drive Constants
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

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

# 0. User Context
st.sidebar.subheader("User Profile")
user_role = st.sidebar.selectbox(
    "Select Your Role",
    ["CEO", "Finance_Manager", "Analyst", "Employee"],
    index=2 # Default to Analyst
)

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

# Session state for Google Drive
if "creds" not in st.session_state:
    st.session_state.creds = None
if "drive_files" not in st.session_state:
    st.session_state.drive_files = []

tab_local, tab_drive = st.sidebar.tabs(["📁 Local Upload", "☁️ Google Drive"])

with tab_local:
    st.markdown("### 📄 Local File Upload")
    col_d, col_s = st.columns(2)
    with col_d:
        domain = st.selectbox("Data Domain", ["finance", "hr", "sales", "operations", "general"], index=0, key="local_domain")
    with col_s:
        sensitivity = st.selectbox("Sensitivity", ["public", "internal", "confidential", "restricted"], index=1, key="local_sens")
    
    uploaded_files = st.file_uploader("Upload Excel Files (Max 50)", type=["xlsx", "xls"], accept_multiple_files=True, key="local_uploader")
    if uploaded_files:
        if len(uploaded_files) > 50:
            st.error("Maximum 50 files allowed. Please remove some.")
        else:
            if st.button("🚀 Process Local Files", use_container_width=True):
                # Ensure metadata tables exist (idempotent)
                conn = get_conn()
                create_metadata_tables(conn)
                conn.close()

                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Wrapper for parallel execution
                def process_single_file(uploaded_file):
                    try:
                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Run ingestion
                        status = process_excel_file(tmp_path, DB_URL, OPENROUTER_KEY, domain, sensitivity, original_filename=uploaded_file.name)
                        
                        os.remove(tmp_path)
                        return {"success": True, "status": status, "name": uploaded_file.name}
                    except Exception as e:
                        return {"success": False, "name": uploaded_file.name, "error": str(e)}

                # Concurrent execution
                completed_count = 0
                total_files = len(uploaded_files)
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(total_files, 8)) as executor:
                    future_to_file = {executor.submit(process_single_file, f): f for f in uploaded_files}
                    
                    for future in concurrent.futures.as_completed(future_to_file):
                        result = future.result()
                        completed_count += 1
                        
                        # Update progress
                        progress = completed_count / total_files
                        progress_bar.progress(progress)
                        
                        if result["success"]:
                            if result.get("status") == "DUPLICATE":
                                st.toast(f"ℹ️ Skipped duplicate: {result['name']}")
                            else:
                                st.toast(f"✅ Processed {result['name']}")
                        else:
                            st.error(f"❌ Error {result['name']}: {result['error']}")
                
                st.success("All files processed!")
                time.sleep(1)
                st.rerun()

with tab_drive:
    st.markdown("### ☁️ Google Drive Source")
    if st.session_state.creds is None:
        st.info("Access files from your Google Drive")
        if st.button("🔑 Login with Google", use_container_width=True):
            try:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "client2.json",
                    SCOPES
                )
                creds = flow.run_local_server(port=0)
                st.session_state.creds = creds
                st.success("✅ Logged in")
                st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
    else:
        # Build service
        service = build("drive", "v3", credentials=st.session_state.creds)
        
        # Fetch files if not already done
        if not st.session_state.drive_files:
            try:
                with st.spinner("Fetching files..."):
                    results = service.files().list(
                        pageSize=30,
                        q="mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' or mimeType='application/vnd.ms-excel'",
                        fields="files(id, name, mimeType)"
                    ).execute()
                    st.session_state.drive_files = results.get("files", [])
            except Exception as e:
                st.error(f"Failed to fetch files: {e}")
        
        if not st.session_state.drive_files:
            st.warning("No Excel files found in Drive.")
            if st.button("🔄 Refresh Files", use_container_width=True):
                st.session_state.drive_files = []
                st.rerun()
        else:
            file_names = [f["name"] for f in st.session_state.drive_files]
            selected_name = st.selectbox("Select a file from Drive", file_names, key="drive_selector")
            
            selected_file = next(f for f in st.session_state.drive_files if f["name"] == selected_name)
            
            col_dd, col_ss = st.columns(2)
            with col_dd:
                g_domain = st.selectbox("Data Domain", ["finance", "hr", "sales", "operations", "general"], index=0, key="g_domain")
            with col_ss:
                g_sensitivity = st.selectbox("Sensitivity", ["public", "internal", "confidential", "restricted"], index=1, key="g_sens")
            
            if st.button("⚡ Ingest from Drive", use_container_width=True):
                conn = get_conn()
                create_metadata_tables(conn)
                conn.close()
                
                with st.spinner(f"Downloading and processing {selected_name}..."):
                    try:
                        # 1. Download from Drive
                        request = service.files().get_media(fileId=selected_file["id"])
                        fh = io.BytesIO()
                        downloader = MediaIoBaseDownload(fh, request)
                        done = False
                        while not done:
                            status, done = downloader.next_chunk()
                        
                        # 2. Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                            tmp_file.write(fh.getvalue())
                            tmp_path = tmp_file.name
                        
                        # 3. Run ingestion (same logic as local)
                        process_excel_file(tmp_path, DB_URL, OPENROUTER_KEY, g_domain, g_sensitivity)
                        
                        st.success(f"Successfully processed {selected_name}")
                        os.remove(tmp_path)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing {selected_name}: {e}")
        
        st.divider()
        if st.button("🚪 Logout from Google", use_container_width=True):
            st.session_state.creds = None
            st.session_state.drive_files = []
            st.rerun()

# Deleted the redundant else block that was part of the radio logic
# if ingest_source == "Local Upload": ... else: ... -> replaced by tabs


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
    s.keywords,
    s.columns_metadata
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
    # 2.5 Business Health Analytics
    # ==========================================
    st.divider()
    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        st.subheader("🏥 Business Health Check")
        st.caption("Automated financial health analysis and trend detection.")
    with col_h2:
        if st.button("🔍 Run Health Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing financial signals..."):
                health_data = compute_business_health(DB_URL, OPENROUTER_KEY, sel_meta)
                st.session_state.health_results = health_data

    if "health_results" in st.session_state and st.session_state.health_results:
        res = st.session_state.health_results
        
        if "error" in res:
            st.error(res["error"])
        elif res.get("status") == "Insufficient Data":
            st.warning(f"**{res['status']}** - {res['reason']}")
            st.info(f"💡 {res['suggested_action']}")
        else:
            # 1. Status Indicator
            status_map = {
                "Healthy": ("✅", "success"),
                "Warning": ("⚠️", "warning"),
                "Risk": ("🚨", "error")
            }
            icon, mode = status_map.get(res["status"], ("ℹ️", "info"))
            
            st.markdown(f"### Status: {icon} {res['status']}")
            
            # 2. Key Metrics
            m = res["metrics"]
            t = res["trends"]
            
            def get_delta(trend):
                return "5%" if trend == "up" else "-5%" if trend == "down" else None
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"${m['revenue']:,.0f}", delta=get_delta(t['revenue']))
            c2.metric("Expenses", f"${m['cost']:,.0f}", delta=get_delta(t['cost']), delta_color="inverse")
            c3.metric("Profit", f"${m['profit']:,.0f}", delta=get_delta(t['profit']))
            c4.metric("Margin", f"{m['margin']}%")
            
            # 3. LLM Summary
            st.info(f"**Analyst Summary:** {res['summary']}")
            
            # 4. Trends Visualization
            if res.get("history"):
                hist_df = pd.DataFrame(res["history"])
                # Map time column (it will be 'index' if we resampled or the original name)
                time_col = 'index' if 'index' in hist_df.columns else hist_df.columns[0]
                
                chart_data = hist_df.melt(id_vars=[time_col], value_vars=["total_revenue", "total_cost", "total_profit"])
                import plotly.express as px
                fig = px.line(chart_data, x=time_col, y="value", color="variable", 
                             title="Financial Trends Over Time",
                             labels={"value": "Amount ($)", "variable": "Metric"})
                st.plotly_chart(fig, use_container_width=True)

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
                result_pack = process_retrieval(user_query, DB_URL, OPENROUTER_KEY, user_role)
                
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
