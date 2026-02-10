"""
Pydantic models for API request/response validation
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


# Request Models

class QueryRequest(BaseModel):
    """Natural language query request"""
    query: str = Field(..., description="Natural language query", min_length=1)
    limit: Optional[int] = Field(3, description="Number of results to return", ge=1, le=10)


class SemanticSearchRequest(BaseModel):
    """Semantic search request"""
    query: str = Field(..., description="Search query", min_length=1)
    threshold: Optional[float] = Field(0.7, description="Similarity threshold", ge=0.0, le=1.0)
    limit: Optional[int] = Field(10, description="Maximum results", ge=1, le=50)


# Response Models

class FileUploadResponse(BaseModel):
    """Response for file upload"""
    success: bool
    task_id: str
    file_id: Optional[str] = None
    file_name: str
    status: str  # "SUCCESS", "DUPLICATE", "PROCESSING"
    message: str
    execution_time: Optional[float] = None


class FileMetadataResponse(BaseModel):
    """File metadata response"""
    file_id: str
    file_name: str
    uploaded_at: datetime
    num_sheets: Optional[int] = None
    num_pages: Optional[int] = None
    summary: Optional[str] = None
    keywords: Optional[str] = None
    file_hash: Optional[str] = None


class SheetMetadataResponse(BaseModel):
    """Sheet metadata response"""
    sheet_id: str
    file_id: str
    sheet_name: str
    table_name: str
    num_rows: int
    num_columns: int
    category: Optional[str] = None
    summary: Optional[str] = None
    keywords: Optional[str] = None
    columns_metadata: Optional[Dict[str, Any]] = None


class FileDetailResponse(BaseModel):
    """Detailed file response with sheets"""
    file: FileMetadataResponse
    sheets: List[SheetMetadataResponse]


class QueryResponse(BaseModel):
    """Query response with results"""
    success: bool
    answer: str
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    sources: Optional[List[Dict[str, str]]] = None
    chart_available: bool = False


class SearchResultItem(BaseModel):
    """Single search result"""
    file_name: str
    sheet_name: Optional[str] = None
    summary: str
    similarity: float
    file_id: str
    sheet_id: Optional[str] = None


class SearchResponse(BaseModel):
    """Search results response"""
    success: bool
    query: str
    results: List[SearchResultItem]
    total_results: int


class StatsResponse(BaseModel):
    """System statistics response"""
    total_files: int
    total_sheets: int
    total_rows: Optional[int] = None
    total_pdf_chunks: Optional[int] = None
    storage_info: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    database_connected: bool
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchUploadResponse(BaseModel):
    """Batch upload response"""
    success: bool
    total_files: int
    successful: int
    failed: int
    results: List[FileUploadResponse]
