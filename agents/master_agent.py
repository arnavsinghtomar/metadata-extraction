"""
Master Orchestrator Agent - Central coordinator for all agents
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime

from agents.base_agent import BaseAgent, AgentStatus
from core.schemas import (
    AgentMessage, TaskType, TaskStatus,
    IngestionRequest, IngestionResponse,
    QueryRequest, QueryResponse,
    MaintenanceRequest, MaintenanceResponse
)
from core.config import get_config
from core.database import DatabasePool
from utils.logger import get_logger
from utils.helpers import generate_id


class MasterAgent(BaseAgent):
    """
    Master Orchestrator Agent.
    
    Responsibilities:
    - Route requests to appropriate agents
    - Manage agent lifecycle
    - Handle errors and retries
    - Coordinate multi-agent workflows
    """
    
    def __init__(self):
        super().__init__(name="MasterOrchestrator")
        self.config = get_config()
        self.db_pool = DatabasePool(
            self.config.database_url,
            min_conn=self.config.db_pool_min,
            max_conn=self.config.db_pool_max
        )
        
        # Registry of available agents (lazy loaded)
        self._agents = {}
        
        self.log_info("Master Orchestrator Agent initialized")
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task by routing to appropriate agent.
        
        Args:
            task: Task dictionary with 'type' and 'payload'
            
        Returns:
            Result dictionary
        """
        self._update_status(AgentStatus.PROCESSING)
        start_time = time.time()
        
        try:
            task_type = task.get('type')
            payload = task.get('payload', {})
            
            self.log_info(f"Routing task: {task_type}")
            
            # Route to appropriate handler
            if task_type == TaskType.INGESTION:
                result = self._handle_ingestion(payload)
            elif task_type == TaskType.QUERY:
                result = self._handle_query(payload)
            elif task_type == TaskType.MAINTENANCE:
                result = self._handle_maintenance(payload)
            elif task_type == TaskType.HEALTH_CHECK:
                result = self._handle_health_check()
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            self._update_status(AgentStatus.COMPLETED)
            
            result['processing_time_seconds'] = time.time() - start_time
            return result
            
        except Exception as e:
            return self.handle_error(e, {"task": task})
    
    def _handle_ingestion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle file ingestion request.
        
        Args:
            payload: Ingestion request payload
            
        Returns:
            Ingestion result
        """
        self.log_info("Delegating to Ingestion Agent")
        
        # Lazy load Ingestion Agent
        if 'ingestion' not in self._agents:
            from agents.ingestion_agent import IngestionAgent
            self._agents['ingestion'] = IngestionAgent()
        
        ingestion_agent = self._agents['ingestion']
        
        # Execute with retry logic
        return self._execute_with_retry(ingestion_agent, payload)
    
    def _handle_query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle query request.
        
        Args:
            payload: Query request payload
            
        Returns:
            Query result
        """
        self.log_info("Delegating to Query Agent")
        
        # Lazy load Query Agent
        if 'query' not in self._agents:
            from agents.query_agent import QueryAgent
            self._agents['query'] = QueryAgent()
        
        query_agent = self._agents['query']
        
        # Execute with retry logic
        return self._execute_with_retry(query_agent, payload)
    
    def _handle_maintenance(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle maintenance request.
        
        Args:
            payload: Maintenance request payload
            
        Returns:
            Maintenance result
        """
        self.log_info("Delegating to Maintenance Agent")
        
        # Lazy load Maintenance Agent
        if 'maintenance' not in self._agents:
            from agents.maintenance_agent import MaintenanceAgent
            self._agents['maintenance'] = MaintenanceAgent()
        
        maintenance_agent = self._agents['maintenance']
        
        return self._execute_with_retry(maintenance_agent, payload)
    
    def _handle_health_check(self) -> Dict[str, Any]:
        """
        Perform system-wide health check.
        
        Returns:
            Health status for all agents
        """
        self.log_info("Performing system health check")
        
        health_status = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "master_agent": self.health_check(),
            "agents": {},
            "database": self._check_database_health()
        }
        
        # Check all loaded agents
        for agent_name, agent in self._agents.items():
            health_status["agents"][agent_name] = agent.health_check()
        
        return health_status
    
    def _check_database_health(self) -> Dict[str, Any]:
        """
        Check database connectivity and stats.
        
        Returns:
            Database health info
        """
        try:
            with self.db_pool.get_cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
            
            return {
                "connected": True,
                "pool_stats": self.db_pool.get_stats()
            }
        except Exception as e:
            self.log_error(f"Database health check failed: {e}")
            return {
                "connected": False,
                "error": str(e)
            }
    
    def _execute_with_retry(
        self,
        agent: BaseAgent,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute agent task with retry logic.
        
        Args:
            agent: Agent to execute
            payload: Task payload
            
        Returns:
            Execution result
        """
        max_retries = self.config.max_retries
        retry_delay = self.config.retry_delay_seconds
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                self.log_info(f"Executing {agent.name} (attempt {attempt + 1}/{max_retries})")
                result = agent.execute(payload)
                
                # Check if execution was successful
                if result.get('success', False):
                    return result
                
                # If not successful but no exception, treat as error
                last_error = result.get('error', 'Unknown error')
                self.log_warning(f"Attempt {attempt + 1} failed: {last_error}")
                
            except Exception as e:
                last_error = str(e)
                self.log_error(f"Attempt {attempt + 1} raised exception: {last_error}")
            
            # Wait before retry (except on last attempt)
            if attempt < max_retries - 1:
                self.log_info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        
        # All retries exhausted
        return {
            "success": False,
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
            "agent": agent.name
        }
    
    def route_task(self, task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Public API to route a task.
        
        Args:
            task_type: Type of task (ingestion, query, maintenance)
            payload: Task payload
            
        Returns:
            Task result
        """
        task = {
            'type': task_type,
            'payload': payload
        }
        return self.execute(task)


# Global master agent instance
_master_agent: Optional[MasterAgent] = None


def get_master_agent() -> MasterAgent:
    """
    Get the global master agent instance.
    
    Returns:
        MasterAgent instance
    """
    global _master_agent
    if _master_agent is None:
        _master_agent = MasterAgent()
    return _master_agent
