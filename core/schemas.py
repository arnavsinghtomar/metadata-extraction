"""
Pydantic schemas for agent communication and data validation
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class TaskType(str, Enum):
    """Types of tasks that can be executed"""
    INGESTION = "ingestion"
    QUERY = "query"
    MAINTENANCE = "maintenance"
    HEALTH_CHECK = "health_check"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ==================== Base Message Schema ====================

class AgentMessage(BaseModel):
    """Base message format for agent communication"""
    agent_id: str = Field(..., description="ID of the agent handling this message")
    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task payload")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message if failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    
    class Config:
        use_enum_values = True


# ==================== Ingestion Schemas ====================

class IngestionRequest(BaseModel):
    """Request schema for file ingestion"""
    file_path: str = Field(..., description="Path to the Excel file")
    file_name: str = Field(..., description="Original file name")
    db_url: str = Field(..., description="Database connection URL")
    openai_key: str = Field(..., description="OpenAI API key")
    
    class Config:
        schema_extra = {
            "example": {
                "file_path": "/tmp/financial_data.xlsx",
                "file_name": "financial_data.xlsx",
                "db_url": "postgresql://user:pass@localhost/db",
                "openai_key": "sk-..."
            }
        }


class SheetMetadata(BaseModel):
    """Metadata for a single sheet"""
    sheet_id: str
    sheet_name: str
    table_name: str
    num_rows: int
    num_columns: int
    category: Optional[str] = None
    summary: Optional[str] = None
    keywords: Optional[str] = None


class IngestionResponse(BaseModel):
    """Response schema for file ingestion"""
    success: bool
    file_id: str
    file_name: str
    num_sheets: int
    sheets_processed: List[SheetMetadata]
    file_summary: Optional[str] = None
    file_keywords: Optional[str] = None
    error: Optional[str] = None
    processing_time_seconds: float


# ==================== Query Schemas ====================

class QueryRequest(BaseModel):
    """Request schema for natural language query"""
    query: str = Field(..., description="User's natural language query")
    db_url: str = Field(..., description="Database connection URL")
    openai_key: str = Field(..., description="OpenAI API key")
    max_results: int = Field(default=3, description="Max number of sheets to search")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What was the total revenue in 2024?",
                "db_url": "postgresql://user:pass@localhost/db",
                "openai_key": "sk-...",
                "max_results": 3
            }
        }


class SheetMatch(BaseModel):
    """Schema for matched sheet in retrieval"""
    sheet_id: str
    table_name: str
    sheet_name: str
    category: Optional[str]
    distance: float
    columns_metadata: Optional[List[Dict[str, Any]]] = None


class QueryResponse(BaseModel):
    """Response schema for query execution"""
    success: bool
    query: str
    final_answer: str
    confidence_score: Optional[float] = None
    sheet_matches: List[SheetMatch] = []
    generated_sql: Optional[str] = None
    results_df_json: Optional[str] = None  # DataFrame as JSON string
    debug_log: List[str] = []
    error: Optional[str] = None
    processing_time_seconds: float


# ==================== Maintenance Schemas ====================

class MaintenanceTask(str, Enum):
    """Types of maintenance tasks"""
    CLEANUP = "cleanup"
    OPTIMIZE = "optimize"
    MONITOR = "monitor"
    HEALTH_CHECK = "health_check"


class MaintenanceRequest(BaseModel):
    """Request schema for maintenance tasks"""
    task: MaintenanceTask
    db_url: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MaintenanceResponse(BaseModel):
    """Response schema for maintenance tasks"""
    success: bool
    task: MaintenanceTask
    details: Dict[str, Any]
    error: Optional[str] = None
    processing_time_seconds: float


# ==================== Health Check Schemas ====================

class AgentHealthStatus(BaseModel):
    """Health status for a single agent"""
    agent_id: str
    agent_name: str
    status: str
    uptime_seconds: float
    last_activity: datetime


class SystemHealthResponse(BaseModel):
    """Overall system health response"""
    healthy: bool
    agents: List[AgentHealthStatus]
    database_connected: bool
    timestamp: datetime = Field(default_factory=datetime.now)
