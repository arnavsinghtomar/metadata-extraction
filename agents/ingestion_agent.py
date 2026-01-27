"""
Ingestion Agent - Handles file processing and data extraction
"""

import os
import time
from typing import Optional
from datetime import datetime
from pydantic import Field

from .base_agent import BaseAgent, AgentTask, AgentResponse, TaskStatus, AgentCapability
from ingest_excel import process_excel_file, create_metadata_tables, get_db_connection


class IngestionAgent(BaseAgent):
    """
    Agent responsible for ingesting files into the database
    Supports Excel, PDF, and CSV files
    """
    
    name: str = Field(default="IngestionAgent")
    description: str = Field(
        default="Processes and ingests files (Excel, PDF, CSV) into the database with metadata extraction"
    )
    capabilities: list = Field(
        default_factory=lambda: [
            AgentCapability(
                name="excel_ingestion",
                description="Process Excel files with multiple sheets",
                required_tools=["ingest_excel.py", "openpyxl", "pandas"]
            ),
            AgentCapability(
                name="pdf_ingestion",
                description="Extract and process PDF documents",
                required_tools=["ingest_pdf.py", "pypdf"]
            ),
            AgentCapability(
                name="csv_ingestion",
                description="Process CSV and structured data files",
                required_tools=["ingest_structured.py", "pandas"]
            ),
            AgentCapability(
                name="metadata_extraction",
                description="Extract metadata using AI",
                required_tools=["openai", "pgvector"]
            )
        ]
    )
    
    db_url: Optional[str] = Field(default=None)
    openai_key: Optional[str] = Field(default=None)
    
    def __init__(self, db_url: str, openai_key: str, **data):
        super().__init__(**data)
        self.db_url = db_url
        self.openai_key = openai_key
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Ensure database schema exists"""
        try:
            conn = get_db_connection(self.db_url)
            create_metadata_tables(conn)
            conn.close()
        except Exception as e:
            print(f"Warning: Could not initialize schema: {e}")
    
    def validate_task(self, task: AgentTask) -> bool:
        """Validate ingestion task"""
        if task.task_type != "ingest":
            return False
        
        required_fields = ["file_path"]
        return all(field in task.payload for field in required_fields)
    
    def execute(self, task: AgentTask) -> AgentResponse:
        """
        Execute file ingestion task
        
        Expected task.payload:
            - file_path: str (path to file)
            - original_filename: str (optional, for uploaded files)
        """
        start_time = time.time()
        
        try:
            # Validate task
            if not self.validate_task(task):
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error="Invalid task: missing required fields (file_path)",
                    completed_at=datetime.now()
                )
            
            file_path = task.payload["file_path"]
            original_filename = task.payload.get("original_filename")
            
            # Check file exists
            if not os.path.exists(file_path):
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"File not found: {file_path}",
                    completed_at=datetime.now()
                )
            
            # Determine file type and process
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in ['.xlsx', '.xls']:
                result = self._process_excel(file_path, original_filename)
            elif file_ext == '.pdf':
                result = self._process_pdf(file_path, original_filename)
            elif file_ext == '.csv':
                result = self._process_csv(file_path, original_filename)
            else:
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Unsupported file type: {file_ext}",
                    completed_at=datetime.now()
                )
            
            execution_time = time.time() - start_time
            
            response = AgentResponse(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                completed_at=datetime.now(),
                metadata={
                    "file_type": file_ext,
                    "file_path": file_path
                }
            )
            
            self.log_execution(task, response)
            return response
            
        except Exception as e:
            return self.handle_error(task, e)
    
    def _process_excel(self, file_path: str, original_filename: Optional[str] = None) -> dict:
        """Process Excel file"""
        status = process_excel_file(
            file_path, 
            self.db_url, 
            self.openai_key,
            original_filename=original_filename
        )
        
        return {
            "status": status,
            "file_type": "excel",
            "message": "Excel file processed successfully" if status == "SUCCESS" else "File was duplicate"
        }
    
    def _process_pdf(self, file_path: str, original_filename: Optional[str] = None) -> dict:
        """Process PDF file"""
        from ingest_pdf import process_pdf_file
        
        process_pdf_file(file_path, self.db_url, self.openai_key)
        
        return {
            "status": "SUCCESS",
            "file_type": "pdf",
            "message": "PDF file processed successfully"
        }
    
    def _process_csv(self, file_path: str, original_filename: Optional[str] = None) -> dict:
        """Process CSV file"""
        from ingest_structured import process_structured_file
        
        process_structured_file(
            file_path, 
            self.db_url, 
            self.openai_key,
            original_filename=original_filename
        )
        
        return {
            "status": "SUCCESS",
            "file_type": "csv",
            "message": "CSV file processed successfully"
        }
