"""
Query Agent - Handles natural language queries using RAG
"""

import time
from typing import Optional
from datetime import datetime
from pydantic import Field

from .base_agent import BaseAgent, AgentTask, AgentResponse, TaskStatus, AgentCapability
from retrieval import process_retrieval


class QueryAgent(BaseAgent):
    """
    Agent responsible for answering natural language queries
    Uses RAG (Retrieval-Augmented Generation) with vector search
    """
    
    name: str = Field(default="QueryAgent")
    description: str = Field(
        default="Answers natural language queries using semantic search and SQL generation"
    )
    capabilities: list = Field(
        default_factory=lambda: [
            AgentCapability(
                name="semantic_search",
                description="Find relevant data using vector similarity",
                required_tools=["pgvector", "embeddings"]
            ),
            AgentCapability(
                name="sql_generation",
                description="Generate SQL queries from natural language",
                required_tools=["openai", "retrieval.py"]
            ),
            AgentCapability(
                name="result_synthesis",
                description="Synthesize natural language answers from query results",
                required_tools=["openai"]
            ),
            AgentCapability(
                name="visualization",
                description="Generate charts for query results",
                required_tools=["plotly"]
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
        """Validate query task"""
        if task.task_type != "query":
            return False
        
        required_fields = ["question"]
        return all(field in task.payload for field in required_fields)
    
    def execute(self, task: AgentTask) -> AgentResponse:
        """
        Execute query task
        
        Expected task.payload:
            - question: str (natural language query)
        """
        start_time = time.time()
        
        try:
            # Validate task
            if not self.validate_task(task):
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error="Invalid task: missing required field (question)",
                    completed_at=datetime.now()
                )
            
            question = task.payload["question"]
            
            # Process query using RAG pipeline
            result_pack = process_retrieval(question, self.db_url, self.openai_key)
            
            # Check for errors
            if "error" in result_pack:
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=result_pack["error"],
                    completed_at=datetime.now()
                )
            
            execution_time = time.time() - start_time
            
            # Format response
            response = AgentResponse(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result={
                    "answer": result_pack.get("final_answer"),
                    "sql_query": result_pack.get("generated_sql"),
                    "source_sheet": result_pack.get("sheet_match", {}).get("sheet_name"),
                    "source_file": result_pack.get("sheet_match", {}).get("file_name"),
                    "has_chart": "chart" in result_pack,
                    "chart_spec": result_pack.get("chart_spec"),
                    "results_preview": result_pack.get("results_df").head(5).to_dict() if "results_df" in result_pack else None
                },
                execution_time=execution_time,
                completed_at=datetime.now(),
                metadata={
                    "question": question,
                    "match_confidence": 1 - result_pack.get("sheet_match", {}).get("distance", 1)
                }
            )
            
            self.log_execution(task, response)
            return response
            
        except Exception as e:
            return self.handle_error(task, e)
