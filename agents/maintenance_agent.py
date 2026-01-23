"""
Maintenance Agent - Handles database maintenance tasks
"""

from typing import Dict, Any
from agents.base_agent import BaseAgent, AgentStatus
from core.config import get_config
from core.database import DatabasePool


class MaintenanceAgent(BaseAgent):
    """
    Maintenance Agent for database cleanup and maintenance.
    
    Responsibilities:
    - Clean up database
    - Vacuum and optimize
    - Remove orphaned records
    """
    
    def __init__(self):
        super().__init__(name="MaintenanceAgent")
        self.config = get_config()
        self.db_pool = DatabasePool(
            self.config.database_url,
            min_conn=self.config.db_pool_min,
            max_conn=self.config.db_pool_max
        )
        self.log_info("Maintenance Agent initialized")
    
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute maintenance task.
        
        Expected payload:
        {
            "action": str,  # cleanup, vacuum, etc.
        }
        
        Returns:
            Maintenance result
        """
        self._update_status(AgentStatus.PROCESSING)
        
        try:
            action = payload.get('action', 'cleanup')
            
            self.log_info(f"Performing maintenance action: {action}")
            
            if action == 'cleanup':
                return self._cleanup_database()
            else:
                raise ValueError(f"Unknown maintenance action: {action}")
            
        except Exception as e:
            return self.handle_error(e, {"payload": payload})
    
    def _cleanup_database(self) -> Dict[str, Any]:
        """Clean up database by removing all data"""
        try:
            with self.db_pool.get_cursor(commit=True) as cur:
                # Get all dynamic tables (sheet_*)
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE 'sheet_%'
                """)
                tables = [row[0] for row in cur.fetchall()]
                
                # Drop all dynamic tables
                for table in tables:
                    cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                    self.log_info(f"Dropped table: {table}")
                
                # Clear metadata tables
                cur.execute("DELETE FROM sheets_metadata")
                cur.execute("DELETE FROM files_metadata")
                
                self.log_info("Database cleanup completed")
            
            self._update_status(AgentStatus.COMPLETED)
            
            return {
                "success": True,
                "message": "Database cleaned successfully",
                "tables_dropped": len(tables)
            }
            
        except Exception as e:
            return self.handle_error(e, {"action": "cleanup"})
