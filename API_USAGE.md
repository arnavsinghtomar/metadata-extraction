# Metadata Ingestion API - Usage Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Ensure your `.env` file has:
```env
DATABASE_URL=postgresql://user:pass@host/db
OPENROUTER_API_KEY=your_openai_key
```

### 3. Start the API Server

```bash
# Development mode (auto-reload)
uvicorn api_main:app --reload --port 8000

# Production mode
uvicorn api_main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📚 API Endpoints

### File Ingestion

#### Upload Single File
```bash
POST /api/v1/ingest/file
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@financial_report.xlsx"
```

**Response:**
```json
{
  "success": true,
  "task_id": "123e4567-e89b-12d3-a456-426614174000",
  "file_name": "financial_report.xlsx",
  "status": "SUCCESS",
  "message": "Excel file processed successfully",
  "execution_time": 12.5
}
```

---

#### Upload Multiple Files (Batch)
```bash
POST /api/v1/ingest/batch
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/batch" \
  -F "files=@file1.xlsx" \
  -F "files=@file2.pdf" \
  -F "files=@file3.csv"
```

**Response:**
```json
{
  "success": true,
  "total_files": 3,
  "successful": 2,
  "failed": 1,
  "results": [...]
}
```

---

### Metadata Retrieval

#### List All Files
```bash
GET /api/v1/files?limit=10&offset=0
```

**Example:**
```bash
curl "http://localhost:8000/api/v1/files?limit=10"
```

**Response:**
```json
[
  {
    "file_id": "123e4567-e89b-12d3-a456-426614174000",
    "file_name": "financial_report.xlsx",
    "uploaded_at": "2026-02-09T10:00:00",
    "num_sheets": 3,
    "summary": "Financial data for Q4 2023",
    "keywords": "revenue, expenses, profit"
  }
]
```

---

#### Get File Details
```bash
GET /api/v1/files/{file_id}
```

**Example:**
```bash
curl "http://localhost:8000/api/v1/files/123e4567-e89b-12d3-a456-426614174000"
```

**Response:**
```json
{
  "file": {
    "file_id": "...",
    "file_name": "financial_report.xlsx",
    ...
  },
  "sheets": [
    {
      "sheet_id": "...",
      "sheet_name": "Revenue",
      "num_rows": 100,
      "num_columns": 5,
      "category": "Financial Data",
      "summary": "Monthly revenue breakdown"
    }
  ]
}
```

---

#### Get Sheet Metadata
```bash
GET /api/v1/sheets/{sheet_id}
```

---

#### Delete File
```bash
DELETE /api/v1/files/{file_id}
```

**Example:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/files/123e4567-e89b-12d3-a456-426614174000"
```

---

### Query & Search

#### Natural Language Query
```bash
POST /api/v1/query
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the total revenue in 2023?",
    "limit": 3
  }'
```

**Response:**
```json
{
  "success": true,
  "answer": "The total revenue in 2023 was $1.2M based on the Revenue sheet.",
  "sql": "SELECT SUM(amount) FROM revenue_2023 WHERE year = 2023",
  "data": [
    {"total": 1200000}
  ],
  "sources": [
    {
      "file_name": "financial_report.xlsx",
      "sheet_name": "Revenue"
    }
  ],
  "chart_available": false
}
```

---

#### Semantic Search
```bash
POST /api/v1/search/semantic
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/search/semantic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sales performance",
    "threshold": 0.7,
    "limit": 5
  }'
```

**Response:**
```json
{
  "success": true,
  "query": "sales performance",
  "results": [
    {
      "file_name": "Q4_report.xlsx",
      "sheet_name": "Sales",
      "summary": "Quarterly sales performance metrics",
      "similarity": 0.89,
      "file_id": "...",
      "sheet_id": "..."
    }
  ],
  "total_results": 3
}
```

---

### System

#### Health Check
```bash
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database_connected": true,
  "timestamp": "2026-02-09T10:00:00"
}
```

---

#### System Statistics
```bash
GET /api/v1/stats
```

**Response:**
```json
{
  "total_files": 25,
  "total_sheets": 78,
  "total_rows": 15000,
  "total_pdf_chunks": 450
}
```

---

## 🐍 Python Client Example

```python
import requests

API_BASE = "http://localhost:8000"

# Upload file
with open("data.xlsx", "rb") as f:
    response = requests.post(
        f"{API_BASE}/api/v1/ingest/file",
        files={"file": f}
    )
    print(response.json())

# Query data
response = requests.post(
    f"{API_BASE}/api/v1/query",
    json={"query": "Show me revenue trends"}
)
print(response.json()["answer"])

# List files
response = requests.get(f"{API_BASE}/api/v1/files")
files = response.json()
for file in files:
    print(f"{file['file_name']} - {file['num_sheets']} sheets")
```

---

## 🔧 Configuration

### File Upload Limits
- **Max file size**: 50MB
- **Allowed types**: `.xlsx`, `.xls`, `.pdf`, `.csv`
- **Batch limit**: 20 files per request

### Concurrent Processing
- Batch uploads use ThreadPoolExecutor with max 8 workers
- Adjust in `api/routers/ingestion.py` if needed

---

## 🚢 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api_main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t metadata-api .
docker run -p 8000:8000 --env-file .env metadata-api
```

---

### Production Considerations

1. **Add authentication** (JWT, API keys)
2. **Rate limiting** (slowapi)
3. **HTTPS/SSL** (nginx reverse proxy)
4. **Monitoring** (Prometheus, Grafana)
5. **Logging** (structured JSON logs)

---

## 🧪 Testing

### Using curl
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Upload test file
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@test.xlsx"
```

### Using Python
```python
import requests

# Test upload
files = {"file": open("test.xlsx", "rb")}
r = requests.post("http://localhost:8000/api/v1/ingest/file", files=files)
assert r.status_code == 200
```

---

## 📝 Notes

- All endpoints return JSON responses
- File IDs and Sheet IDs are UUIDs
- Timestamps are in ISO 8601 format
- Vector embeddings use OpenAI's 3072-dimensional model
- Database uses PostgreSQL with pgvector extension

---

## 🆘 Troubleshooting

### API won't start
- Check `.env` file exists with correct variables
- Verify database connection
- Check port 8000 is not in use

### File upload fails
- Check file size (< 50MB)
- Verify file extension is allowed
- Check database has space

### Query returns no results
- Ensure files are ingested successfully
- Check database has metadata
- Try lowering similarity threshold

---

## 📞 Support

For issues or questions, check:
- API docs: http://localhost:8000/docs
- Logs: Check console output
- Database: Verify tables exist and have data
