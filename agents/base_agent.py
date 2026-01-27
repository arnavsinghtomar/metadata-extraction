"""
Base Agent class using Pydantic for type safety and validation
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentTask(BaseModel):
    """Pydantic model for agent tasks"""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type of task (e.g., 'ingest', 'query', 'analyze')")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task-specific data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_123",
                "task_type": "ingest",
                "payload": {"file_path": "/path/to/file.xlsx"},
                "metadata": {"user_id": "user_1"}
            }
        }


class AgentResponse(BaseModel):
    """Pydantic model for agent responses"""
    task_id: str = Field(..., description="Task identifier")
    status: TaskStatus = Field(..., description="Execution status")
    result: Optional[Any] = Field(None, description="Task result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    completed_at: Optional[datetime] = Field(None)
    
    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task_123",
                "status": "completed",
                "result": {"file_id": "abc-123", "sheets_processed": 5},
                "execution_time": 12.5
            }
        }


class AgentCapability(BaseModel):
    """Agent capability definition"""
    name: str = Field(..., description="Capability name")
    description: str = Field(..., description="What this capability does")
    required_tools: List[str] = Field(default_factory=list, description="Required tools/modules")


class BaseAgent(BaseModel):
    """
    Base Agent class with Pydantic validation
    All agents inherit from this class
    """
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: List[AgentCapability] = Field(default_factory=list)
    enabled: bool = Field(default=True, description="Whether agent is enabled")
    
    class Config:
        arbitrary_types_allowed = True
    
    def execute(self, task: AgentTask) -> AgentResponse:
        """
        Execute a task. Must be implemented by subclasses.
        
        Args:
            task: AgentTask object with task details
            
        Returns:
            AgentResponse with execution results
        """
        raise NotImplementedError("Subclasses must implement execute()")
    
    def validate_task(self, task: AgentTask) -> bool:
        """
        Validate if this agent can handle the task
        
        Args:
            task: AgentTask to validate
            
        Returns:
            True if agent can handle this task
        """
        return True
    
    def log_execution(self, task: AgentTask, response: AgentResponse):
        """Log task execution"""
        logger.info(
            f"[{self.name}] Task {task.task_id} - "
            f"Status: {response.status} - "
            f"Time: {response.execution_time:.2f}s"
        )
    
    def handle_error(self, task: AgentTask, error: Exception) -> AgentResponse:
        """
        Handle execution errors
        
        Args:
            task: Failed task
            error: Exception that occurred
            
        Returns:
            AgentResponse with error details
        """
        logger.error(f"[{self.name}] Task {task.task_id} failed: {str(error)}")
        return AgentResponse(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            error=str(error),
            completed_at=datetime.now()
        )
