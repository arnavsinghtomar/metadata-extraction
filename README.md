# 📊 Financial Data Extraction & Analysis System

AI-powered financial data extraction and analysis platform with multi-agent architecture, RAG-based querying, and automated business health monitoring.

## ✨ Features

### 🤖 Multi-Agent System
- **Master Agent**: Intelligent task routing and orchestration
- **Ingestion Agent**: Process Excel, PDF, and CSV files
- **Query Agent**: Natural language queries using RAG
- **Analytics Agent**: Business health analysis and insights
- **Maintenance Agent**: Database management and optimization

### 📈 Core Capabilities
- **File Processing**: Excel, PDF, CSV with AI-powered metadata extraction
- **Semantic Search**: Vector-based search using OpenAI embeddings
- **SQL Generation**: Natural language to SQL conversion
- **Business Analytics**: Automated financial health checks
- **Trend Detection**: Revenue, cost, and profit trend analysis
- **Interactive Charts**: Plotly visualizations

### 🔍 Advanced Features
- **Google Drive Integration**: Direct file upload from Google Drive
- **Duplicate Detection**: Content-based file versioning
- **Multi-sheet Processing**: Parallel processing for faster ingestion
- **Type-Safe Agents**: Pydantic-based validation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database (Neon recommended)
- OpenRouter API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/metadata-extraction-new.git
cd metadata-extraction-new
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

4. Run the application:
```bash
streamlit run app.py
```

## 🔧 Configuration

Create a `.env` file with:

```env
DATABASE_URL=postgresql://user:password@host/database
OPENROUTER_API_KEY=your_openrouter_api_key
```

## 📖 Usage

### File Ingestion
```python
from agents import MasterAgent

master = MasterAgent(db_url=DB_URL, openai_key=API_KEY)

task = master.create_task(
    task_type="ingest",
    payload={"file_path": "data.xlsx"}
)

response = master.execute(task)
```

### Natural Language Queries
```python
task = master.create_task(
    task_type="query",
    payload={"question": "What was total revenue in 2024?"}
)

response = master.execute(task)
print(response.result['answer'])
```

### Business Health Analysis
```python
task = master.create_task(
    task_type="analyze",
    payload={"sheet_info": {...}}
)

response = master.execute(task)
print(response.result['status'])  # Healthy/Warning/Risk
```

## 🏗️ Architecture

```
metadata-extraction-new/
├── agents/                 # Multi-agent system
│   ├── base_agent.py      # Pydantic base classes
│   ├── master_agent.py    # Orchestrator
│   ├── ingestion_agent.py # File processing
│   ├── query_agent.py     # RAG queries
│   ├── analytics_agent.py # Business intelligence
│   └── maintenance_agent.py # Database ops
├── ingest_excel.py        # Excel file processing
├── ingest_pdf.py          # PDF processing
├── ingest_structured.py   # CSV processing
├── retrieval.py           # RAG pipeline
├── analytics.py           # Business health logic
├── cleanup.py             # Database cleanup
├── app.py                 # Streamlit UI
└── requirements.txt       # Dependencies
```

## 🛠️ Tech Stack

- **Backend**: Python, PostgreSQL, pgvector
- **Frontend**: Streamlit
- **AI/ML**: OpenAI (via OpenRouter), embeddings
- **Data Processing**: Pandas, openpyxl, pypdf
- **Visualization**: Plotly
- **Type Safety**: Pydantic
- **Cloud**: Neon Database

## 📊 Database Schema

### files_metadata
- `file_id` (UUID)
- `file_name` (TEXT)
- `uploaded_at` (TIMESTAMP)
- `summary_embedding` (vector 3072)
- `keywords_embedding` (vector 3072)

### sheets_metadata
- `sheet_id` (UUID)
- `file_id` (UUID FK)
- `table_name` (TEXT)
- `summary_embedding` (vector 3072)
- `columns_metadata` (JSONB)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 🐛 Known Issues

- Vector dimension mismatch: Run `python fix_vector_dimensions.py` to fix
- Slow processing: Increase `max_workers` in `ingest_excel.py` line 706

## 🔮 Roadmap

- [ ] Add export agent for report generation
- [ ] Implement notification system
- [ ] Add data validation agent
- [ ] Support for more file formats
- [ ] Advanced anomaly detection
- [ ] Scheduled automated reports

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- OpenRouter for AI API access
- Neon for serverless PostgreSQL
- Streamlit for the amazing framework
