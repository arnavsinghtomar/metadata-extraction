"""
Base Agent class - Abstract base for all agents in the system
"""

import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class AgentStatus(str, Enum):
    """Agent execution status"""
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Provides common functionality:
    - Logging
    - State management
    - Error handling
    - Health checks
    """
    
    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None):
        """
        Initialize base agent.
        
        Args:
            agent_id: Unique identifier for this agent instance
            name: Human-readable name for the agent
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or self.__class__.__name__
        self.status = AgentStatus.IDLE
        self.logger = self._setup_logger()
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup agent-specific logger"""
        logger = logging.getLogger(f"agent.{self.name}.{self.agent_id[:8]}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - [{self.name}] - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's main task.
        
        Args:
            task: Task payload with all necessary parameters
            
        Returns:
            Result dictionary with status and data
        """
        pass
    
    def _update_status(self, status: AgentStatus):
        """Update agent status and last activity timestamp"""
        self.status = status
        self.last_activity = datetime.now()
        self.logger.info(f"Status updated to: {status.value}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check agent health status.
        
        Returns:
            Health status dictionary
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds()
        }
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors with logging and structured response.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            
        Returns:
            Error response dictionary
        """
        self._update_status(AgentStatus.FAILED)
        
        error_msg = str(error)
        self.logger.error(f"Error in {self.name}: {error_msg}", exc_info=True)
        
        return {
            "success": False,
            "error": error_msg,
            "error_type": type(error).__name__,
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(message)
        
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
        
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(message)
