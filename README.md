# 📊 Financial Data Extraction Agent

A robust, AI-powered system for ingesting, analyzing, and querying complex financial Excel data. This application automatically turns messy Excel sheets into structured PostgreSQL tables and allows you to "talk" to your data using natural language.

## 🚀 Key Features

*   **Automated Ingestion**: Upload Excel files with multiple sheets. The system automatically handles header normalization, type inference, and data cleaning.
*   **Dynamic Schema Generation**: Creates a dedicated, strongly-typed PostgreSQL table for *every single sheet* on the fly. No manual `CREATE TABLE` required.
*   **AI Enrichment**: 
    *   **Summaries**: Uses OpenAI `gpt-4o-mini` to generate executive summaries for each sheet and the file as a whole.
    *   **Categorization**: Automatically tags sheets (e.g., "Financial", "HR", "Inventory").
    *   **Vector Embeddings**: Generates embeddings for semantic search capabilities.
*   **Metadata & Schema Tracking**: Captures original column names and sample values to ensure accurate retrieval later.
*   **Chat with Data (RAG)**: A hybrid retrieval engine that allows you to ask questions like *"What was the total revenue in 2024?"*. It uses vector search to find the right sheet and then writes accurate SQL query to fetch the answer.

## 🛠️ Technology Stack

*   **Frontend**: Streamlit
*   **Database**: PostgreSQL + `pgvector` extension
*   **AI/LLM**: OpenAI API (`gpt-4o-mini`, `text-embedding-3-small`)
*   **Data Processing**: Pandas, OpenPyXL, Psycopg2

## ⚙️ Setup & Installation

### 1. Prerequisites
*   Python 3.10+
*   PostgreSQL installed and running.
*   Check that `pgvector` is installed on your Postgres server:
    ```sql
    CREATE EXTENSION vector;
    ```

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
DATABASE_URL=postgresql://user:password@localhost:5432/your_db_name
OPENAI_API_KEY=sk-your-openai-key-here
```

## 🏃‍♂️ Usage

### Running the App
Start the Streamlit interface:
```bash
streamlit run app.py
```

### Workflow
1.  **Ingest**: Go to the sidebar, upload your Excel files (up to 3 at a time), and click "Process Files".
2.  **Explore**: Use the "Database Content" section to browse summarized files and drill down into specific sheets to view raw data.
3.  **Chat**: Scroll to the "**Ask Your Data**" section. Type questions like:
    *   *"Show me the top 5 expenses for Q3."*
    *   *"Compare revenue between 2023 and 2024."*

### Resetting the System
To wipe the database and start fresh, use the **Danger Zone** button in the sidebar or run:
```bash
python cleanup.py
```

## 🧠 Architecture Overview

### The Ingestion Pipeline (`ingest_excel.py`)
1.  **Read**: Pandas loads the Excel sheet.
2.  **Clean**: Empty rows/cols are dropped. Columns are normalized (removing special chars).
3.  **Infer**: Data types are mapped to Postgres types (TEXT, INTEGER, DOUBLE PRECISION).
4.  **Create**: A dynamic table (e.g., `sheet_income_2024_x82a`) is created.
5.  **Enrich**: OpenAI generates a summary and categorizes the sheet. Column metadata (original headers + samples) is stored in a JSONB column in `sheets_metadata`.

### The Retrieval Pipeline (`retrieval.py`)
1.  **Router**: Embeds the user query and searches `sheets_metadata` to find the most relevant sheet.
2.  **SQL Generation**: Feeds the strict schema (including original headers and sample values) to the LLM to generate a safe SQL query.
3.  **Execution**: Runs the query against the specific dynamic table.
4.  **Synthesis**: The LLM translates the raw data results back into a natural language key insight.
