"""
Analytics Agent - Handles business intelligence and insights
"""

import time
from typing import Optional
from datetime import datetime
from pydantic import Field

from .base_agent import BaseAgent, AgentTask, AgentResponse, TaskStatus, AgentCapability
from analytics import compute_business_health


class AnalyticsAgent(BaseAgent):
    """
    Agent responsible for business analytics and insights
    Analyzes financial health, trends, and generates reports
    """
    
    name: str = Field(default="AnalyticsAgent")
    description: str = Field(
        default="Performs business analytics, financial health checks, and trend analysis"
    )
    capabilities: list = Field(
        default_factory=lambda: [
            AgentCapability(
                name="business_health",
                description="Analyze business financial health",
                required_tools=["analytics.py", "openai"]
            ),
            AgentCapability(
                name="trend_detection",
                description="Detect revenue/cost/profit trends",
                required_tools=["pandas", "analytics.py"]
            ),
            AgentCapability(
                name="anomaly_detection",
                description="Identify unusual patterns in financial data",
                required_tools=["pandas", "openai"]
            ),
            AgentCapability(
                name="report_generation",
                description="Generate executive summaries and reports",
                required_tools=["openai", "plotly"]
            )
        ]
    )
    
    db_url: Optional[str] = Field(default=None)
    openai_key: Optional[str] = Field(default=None)
    
    def __init__(self, db_url: str, openai_key: str, **data):
        super().__init__(**data)
        self.db_url = db_url
        self.openai_key = openai_key
    
    def validate_task(self, task: AgentTask) -> bool:
        """Validate analytics task"""
        if task.task_type not in ["analyze", "health_check", "trends"]:
            return False
        
        required_fields = ["sheet_info"]
        return all(field in task.payload for field in required_fields)
    
    def execute(self, task: AgentTask) -> AgentResponse:
        """
        Execute analytics task
        
        Expected task.payload:
            - sheet_info: dict (sheet metadata with table_name, columns_metadata)
            - lookback_periods: int (optional, default 3)
        """
        start_time = time.time()
        
        try:
            # Validate task
            if not self.validate_task(task):
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error="Invalid task: missing required field (sheet_info)",
                    completed_at=datetime.now()
                )
            
            sheet_info = task.payload["sheet_info"]
            lookback_periods = task.payload.get("lookback_periods", 3)
            
            # Compute business health
            health_data = compute_business_health(
                self.db_url,
                self.openai_key,
                sheet_info,
                lookback_periods=lookback_periods
            )
            
            # Check for errors
            if "error" in health_data:
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=health_data["error"],
                    completed_at=datetime.now()
                )
            
            execution_time = time.time() - start_time
            
            # Format response
            response = AgentResponse(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=health_data,
                execution_time=execution_time,
                completed_at=datetime.now(),
                metadata={
                    "sheet_name": sheet_info.get("sheet_name"),
                    "analysis_type": "business_health"
                }
            )
            
            self.log_execution(task, response)
            return response
            
        except Exception as e:
            return self.handle_error(task, e)
