"""
Master Agent - Orchestrates all sub-agents
"""

import uuid
from typing import Dict, Optional
from pydantic import Field
from datetime import datetime

from .base_agent import BaseAgent, AgentTask, AgentResponse, TaskStatus, AgentCapability
from .ingestion_agent import IngestionAgent
from .query_agent import QueryAgent
from .analytics_agent import AnalyticsAgent
from .maintenance_agent import MaintenanceAgent


class MasterAgent(BaseAgent):
    """
    Master Agent that orchestrates all sub-agents
    Routes tasks to appropriate agents based on task type
    """
    
    name: str = Field(default="MasterAgent")
    description: str = Field(
        default="Orchestrates all sub-agents and routes tasks intelligently"
    )
    capabilities: list = Field(
        default_factory=lambda: [
            AgentCapability(
                name="task_routing",
                description="Route tasks to appropriate sub-agents",
                required_tools=[]
            ),
            AgentCapability(
                name="workflow_orchestration",
                description="Coordinate multi-agent workflows",
                required_tools=[]
            ),
            AgentCapability(
                name="agent_management",
                description="Manage and monitor sub-agents",
                required_tools=[]
            )
        ]
    )
    
    sub_agents: Dict[str, BaseAgent] = Field(default_factory=dict)
    
    def __init__(self, db_url: str, openai_key: str, **data):
        super().__init__(**data)
        
        # Initialize sub-agents
        self.sub_agents = {
            "ingestion": IngestionAgent(db_url=db_url, openai_key=openai_key),
            "query": QueryAgent(db_url=db_url, openai_key=openai_key),
            "analytics": AnalyticsAgent(db_url=db_url, openai_key=openai_key),
            "maintenance": MaintenanceAgent(db_url=db_url)
        }
    
    def route_task(self, task: AgentTask) -> str:
        """
        Determine which agent should handle the task
        
        Args:
            task: AgentTask to route
            
        Returns:
            Agent name to handle the task
        """
        task_type = task.task_type.lower()
        
        # Direct routing based on task type
        routing_map = {
            "ingest": "ingestion",
            "upload": "ingestion",
            "process": "ingestion",
            
            "query": "query",
            "search": "query",
            "ask": "query",
            "find": "query",
            
            "analyze": "analytics",
            "health_check": "analytics",
            "trends": "analytics",
            "insights": "analytics",
            
            "cleanup": "maintenance",
            "validate": "maintenance",
            "optimize": "maintenance",
            "maintain": "maintenance"
        }
        
        return routing_map.get(task_type, "query")  # Default to query agent
    
    def execute(self, task: AgentTask) -> AgentResponse:
        """
        Execute task by routing to appropriate sub-agent
        
        Args:
            task: AgentTask to execute
            
        Returns:
            AgentResponse from sub-agent
        """
        try:
            # Route to appropriate agent
            agent_name = self.route_task(task)
            agent = self.sub_agents.get(agent_name)
            
            if not agent:
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"No agent found for task type: {task.task_type}",
                    completed_at=datetime.now()
                )
            
            # Check if agent is enabled
            if not agent.enabled:
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Agent {agent_name} is disabled",
                    completed_at=datetime.now()
                )
            
            # Execute task with sub-agent
            print(f"[MasterAgent] Routing task {task.task_id} to {agent_name}")
            response = agent.execute(task)
            
            # Add routing metadata
            response.metadata["routed_to"] = agent_name
            response.metadata["master_agent"] = self.name
            
            return response
            
        except Exception as e:
            return self.handle_error(task, e)
    
    def create_task(
        self,
        task_type: str,
        payload: dict,
        metadata: Optional[dict] = None
    ) -> AgentTask:
        """
        Helper method to create a new task
        
        Args:
            task_type: Type of task
            payload: Task payload
            metadata: Optional metadata
            
        Returns:
            AgentTask object
        """
        return AgentTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            metadata=metadata or {}
        )
    
    def get_agent_status(self) -> dict:
        """Get status of all sub-agents"""
        return {
            name: {
                "enabled": agent.enabled,
                "capabilities": len(agent.capabilities),
                "description": agent.description
            }
            for name, agent in self.sub_agents.items()
        }
    
    def enable_agent(self, agent_name: str):
        """Enable a sub-agent"""
        if agent_name in self.sub_agents:
            self.sub_agents[agent_name].enabled = True
    
    def disable_agent(self, agent_name: str):
        """Disable a sub-agent"""
        if agent_name in self.sub_agents:
            self.sub_agents[agent_name].enabled = False
