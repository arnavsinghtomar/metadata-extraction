# 📊 Financial Data Extraction Agent

A robust, AI-powered system for ingesting, analyzing, and querying complex financial Excel data. This application automatically turns messy Excel sheets into structured PostgreSQL tables and allows you to "talk" to your data using natural language, with advanced features for bulk processing, cloud integration, and automated analytics.

## 🚀 Key Features

### 1. High-Performance Ingestion
*   **Bulk Processing**: Upload up to **50 Excel files** simultaneously.
*   **Concurrency**: Uses multi-threading to process files and sheets in parallel (up to 40 concurrent database connections).
*   **Versioning & Deduplication**:
    *   **Smart Skipping**: Identical files (same content) are automatically skipped.
    *   **Versioning**: Modified files with the same name are ingested as new versions (v1, v2, etc.).

### 2. Intelligent Data Handling
*   **Automated Schema**: Dynamically creates typed PostgreSQL tables for every sheet.
*   **Normalization**: Handles complex column names, reserved keywords, and leading digits.
*   **AI Enrichment**: Generates summaries, keywords, and categorizations using `gpt-4o-mini`.

### 3. Advanced Analytics & Visualization
*   **Chat with Data (RAG)**:
    *   Ask specific questions: *"What was 2024 revenue?"*
    *   Ask metadata questions: *"Which file contains the vendor list?"*
*   **Smart Charting**: Automatically detects if a query requires visualization and generates interactive **Plotly** charts (Bar, Line, Pie, Scatter).
*   **Business Health Check**: A dedicated module that analyzes your data to flag risks (e.g., "Expenses growing faster than revenue") and calculate profit margins.

### 4. Integrations
*   **Google Drive**: Connect directly to Google Drive to select and ingest Excel files from the cloud.

## 🛠️ Technology Stack

*   **Frontend**: Streamlit
*   **Database**: PostgreSQL (Neon.tech supported) + `pgvector`
*   **AI/LLM**: OpenAI API (`gpt-4o-mini`), Google Gemini Embeddings (`google/gemini-embedding-001`)
*   **Visualization**: Plotly Interactive Charts
*   **Cloud**: Google Drive API

## ⚙️ Setup & Installation

### 1. Prerequisites
*   Python 3.10+
*   PostgreSQL with `vector` extension enabled.

### 2. Clone and Install
```bash
git clone https://github.com/your-repo/financial-data-extraction.git
cd financial-data-extraction

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```bash
DATABASE_URL=postgresql://user:password@host:5432/dbname
OPENROUTER_API_KEY=sk-your-key
```

### 4. Google Drive (Optional)
To use Google Drive integration, place your `client_secret.json` (OAuth 2.0 Credentials) in the root directory.

## 🏃‍♂️ Usage

### Running the App
```bash
streamlit run app.py
```

### Workflow
1.  **Ingest**:
    *   **Local**: Drag & Drop up to 50 files. The system handles them in parallel.
    *   **Drive**: Authenticate and pick files from your drive.
2.  **Health Check**: Click "**Run Health Analysis**" in the sidebar for an instant financial audit.
3.  **Chat**:
    *   *"Show me a line chart of monthly revenue."*
    *   *"Compare Q1 vs Q2 expenses."*
    *   *"Which file has the payroll data?"*

## 🧠 Architecture Highlights
*   **Connection Pooling**: Uses `psycopg2.pool.ThreadedConnectionPool` (Size: 60) to handle high-concurrency ingestion.
*   **Hybrid Search**: Retrieval combines vector similarity (for semantic match) with metadata keyword boosting (for filename matches).
*   **Metadata-First**: Questions about file locations bypass SQL generation and query the metadata layer directly.
