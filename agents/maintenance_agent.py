"""
Maintenance Agent - Handles database operations and cleanup
"""

import time
from typing import Optional
from datetime import datetime
from pydantic import Field

from .base_agent import BaseAgent, AgentTask, AgentResponse, TaskStatus, AgentCapability
from cleanup import cleanup_database


class MaintenanceAgent(BaseAgent):
    """
    Agent responsible for database maintenance and operations
    Handles cleanup, validation, and optimization tasks
    """
    
    name: str = Field(default="MaintenanceAgent")
    description: str = Field(
        default="Performs database maintenance, cleanup, and optimization"
    )
    capabilities: list = Field(
        default_factory=lambda: [
            AgentCapability(
                name="database_cleanup",
                description="Clean up old data and tables",
                required_tools=["cleanup.py", "psycopg2"]
            ),
            AgentCapability(
                name="schema_validation",
                description="Validate database schema integrity",
                required_tools=["psycopg2"]
            ),
            AgentCapability(
                name="data_validation",
                description="Check data quality and consistency",
                required_tools=["pandas", "psycopg2"]
            ),
            AgentCapability(
                name="performance_optimization",
                description="Optimize database performance",
                required_tools=["psycopg2"]
            )
        ]
    )
    
    db_url: Optional[str] = Field(default=None)
    
    def __init__(self, db_url: str, **data):
        super().__init__(**data)
        self.db_url = db_url
    
    def validate_task(self, task: AgentTask) -> bool:
        """Validate maintenance task"""
        return task.task_type in ["cleanup", "validate", "optimize"]
    
    def execute(self, task: AgentTask) -> AgentResponse:
        """
        Execute maintenance task
        
        Expected task.payload:
            - operation: str ("cleanup", "validate", "optimize")
            - force: bool (optional, for cleanup)
        """
        start_time = time.time()
        
        try:
            # Validate task
            if not self.validate_task(task):
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error="Invalid task type for MaintenanceAgent",
                    completed_at=datetime.now()
                )
            
            operation = task.task_type
            
            if operation == "cleanup":
                result = self._cleanup_database()
            elif operation == "validate":
                result = self._validate_database()
            elif operation == "optimize":
                result = self._optimize_database()
            else:
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Unknown operation: {operation}",
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
                    "operation": operation
                }
            )
            
            self.log_execution(task, response)
            return response
            
        except Exception as e:
            return self.handle_error(task, e)
    
    def _cleanup_database(self) -> dict:
        """Clean up database"""
        cleanup_database(self.db_url)
        
        return {
            "status": "SUCCESS",
            "message": "Database cleaned successfully",
            "tables_dropped": "all sheet_* tables",
            "metadata_cleared": True
        }
    
    def _validate_database(self) -> dict:
        """Validate database integrity"""
        from ingest_excel import get_db_connection
        
        conn = get_db_connection(self.db_url)
        cursor = conn.cursor()
        
        # Check metadata tables exist
        cursor.execute("SELECT to_regclass('files_metadata');")
        files_exists = cursor.fetchone()[0] is not None
        
        cursor.execute("SELECT to_regclass('sheets_metadata');")
        sheets_exists = cursor.fetchone()[0] is not None
        
        # Count records
        files_count = 0
        sheets_count = 0
        
        if files_exists:
            cursor.execute("SELECT COUNT(*) FROM files_metadata;")
            files_count = cursor.fetchone()[0]
        
        if sheets_exists:
            cursor.execute("SELECT COUNT(*) FROM sheets_metadata;")
            sheets_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "status": "SUCCESS",
            "metadata_tables_exist": files_exists and sheets_exists,
            "files_count": files_count,
            "sheets_count": sheets_count,
            "is_healthy": files_exists and sheets_exists
        }
    
    def _optimize_database(self) -> dict:
        """Optimize database performance"""
        from ingest_excel import get_db_connection
        
        conn = get_db_connection(self.db_url)
        cursor = conn.cursor()
        
        # Run VACUUM ANALYZE
        conn.autocommit = True
        cursor.execute("VACUUM ANALYZE;")
        
        conn.close()
        
        return {
            "status": "SUCCESS",
            "message": "Database optimized successfully",
            "operations": ["VACUUM", "ANALYZE"]
        }
