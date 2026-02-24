"""
RBAC Agent - Handles role-based access control and policy enforcement
"""

import re
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import Field

from .base_agent import BaseAgent, AgentTask, AgentResponse, TaskStatus, AgentCapability
from ingest_excel import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RBACAgent(BaseAgent):
    """
    Agent responsible for role-based access control
    Validates user permissions and enforces data access policies
    """
    
    name: str = Field(default="RBACAgent")
    description: str = Field(
        default="Manages role-based access control, validates permissions, and enforces data access policies"
    )
    capabilities: list = Field(
        default_factory=lambda: [
            AgentCapability(
                name="access_validation",
                description="Validate user access to data domains and sensitivity levels",
                required_tools=["psycopg2"]
            ),
            AgentCapability(
                name="sql_validation",
                description="Enforce access rules on SQL queries",
                required_tools=["re", "psycopg2"]
            ),
            AgentCapability(
                name="policy_management",
                description="Manage and retrieve retrieval policies",
                required_tools=["psycopg2"]
            ),
            AgentCapability(
                name="permission_check",
                description="Check user permissions for specific operations",
                required_tools=["psycopg2"]
            )
        ]
    )
    
    db_url: Optional[str] = Field(default=None)
    
    def __init__(self, db_url: str, **data):
        super().__init__(**data)
        self.db_url = db_url
        self._ensure_rbac_schema()
    
    def _ensure_rbac_schema(self):
        """Ensure RBAC tables exist"""
        try:
            from rbac import create_rbac_tables
            conn = get_db_connection(self.db_url)
            create_rbac_tables(conn)
            conn.close()
        except Exception as e:
            logger.warning(f"Could not initialize RBAC schema: {e}")
    
    def validate_task(self, task: AgentTask) -> bool:
        """Validate RBAC task"""
        valid_types = ["validate_access", "validate_sql", "check_permission", "get_policies"]
        
        if task.task_type not in valid_types:
            return False
        
        # Different tasks require different fields
        if task.task_type == "validate_sql":
            required_fields = ["sql_query", "sheet_info"]
            return all(field in task.payload for field in required_fields)
        
        if task.task_type == "check_permission":
            required_fields = ["user_role", "data_domain", "sensitivity_level"]
            return all(field in task.payload for field in required_fields)
        
        if task.task_type == "get_policies":
            required_fields = ["user_role"]
            return all(field in task.payload for field in required_fields)
        
        return True
    
    def execute(self, task: AgentTask) -> AgentResponse:
        """
        Execute RBAC task
        
        Expected task.payload:
            For validate_sql:
                - sql_query: str (SQL query to validate)
                - sheet_info: dict (sheet metadata with access rules)
            
            For check_permission:
                - user_role: str (user's role)
                - data_domain: str (data domain to access)
                - sensitivity_level: str (sensitivity level)
            
            For get_policies:
                - user_role: str (user's role)
        """
        start_time = time.time()
        
        try:
            # Validate task
            if not self.validate_task(task):
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error="Invalid task: missing required fields",
                    completed_at=datetime.now()
                )
            
            # Route to appropriate handler
            if task.task_type == "validate_sql":
                result = self._validate_sql_access(
                    task.payload["sql_query"],
                    task.payload["sheet_info"]
                )
            elif task.task_type == "check_permission":
                result = self._check_user_permission(
                    task.payload["user_role"],
                    task.payload["data_domain"],
                    task.payload["sensitivity_level"]
                )
            elif task.task_type == "get_policies":
                result = self._get_user_policies(
                    task.payload["user_role"]
                )
            else:
                return AgentResponse(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Unknown task type: {task.task_type}",
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
                    "task_type": task.task_type
                }
            )
            
            self.log_execution(task, response)
            return response
            
        except PermissionError as e:
            # Permission errors are expected and should be returned as failed tasks
            return AgentResponse(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                completed_at=datetime.now(),
                metadata={"error_type": "PermissionError"}
            )
        except Exception as e:
            return self.handle_error(task, e)
    
    def _validate_sql_access(self, sql_query: str, sheet_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runtime validation to enforce access rules on SQL queries
        
        Args:
            sql_query: SQL query to validate
            sheet_info: Sheet metadata with access control info
            
        Returns:
            Dict with validation result
            
        Raises:
            PermissionError: If access is denied
        """
        allow_raw = sheet_info.get('allow_raw_rows', False)
        
        if not allow_raw:
            sql_lower = sql_query.lower()
            
            # Check for forbidden patterns (row-level access)
            forbidden_patterns = [
                r'select\s+\*',
                r'limit',  # often used to peek at rows
                r'offset',
            ]
            
            for pattern in forbidden_patterns:
                if re.search(pattern, sql_lower):
                    raise PermissionError(
                        f"🔒 Access Denied: Raw row-level queries not allowed for {sheet_info.get('data_domain', 'this')} data. "
                        f"Only aggregated summaries are permitted."
                    )
            
            # Must contain at least one aggregation function or a GROUP BY
            required_patterns = [r'sum\(', r'avg\(', r'count\(', r'max\(', r'min\(', r'group\s+by']
            
            has_aggregation = any(re.search(pattern, sql_lower) for pattern in required_patterns)
            
            if not has_aggregation:
                raise PermissionError(
                    f"🔒 Access Denied: You must use aggregation functions (SUM, COUNT, AVG) for this data."
                )
        
        return {
            "valid": True,
            "sql_query": sql_query,
            "access_level": "raw" if allow_raw else "aggregated",
            "message": "SQL query validated successfully"
        }
    
    def _check_user_permission(self, user_role: str, data_domain: str, sensitivity_level: str) -> Dict[str, Any]:
        """
        Check if user has permission to access specific data
        
        Args:
            user_role: User's role (e.g., CEO, Analyst, Employee)
            data_domain: Data domain (e.g., finance, hr, sales)
            sensitivity_level: Sensitivity level (e.g., public, internal, confidential, restricted)
            
        Returns:
            Dict with permission details
        """
        conn = get_db_connection(self.db_url)
        
        try:
            cur = conn.cursor()
            
            query = """
                SELECT allowed, allow_aggregation, allow_raw_rows
                FROM retrieval_policies
                WHERE role = %s AND data_domain = %s AND sensitivity_level = %s
            """
            
            cur.execute(query, (user_role, data_domain, sensitivity_level))
            result = cur.fetchone()
            
            if result is None:
                # No policy found - default to deny
                return {
                    "allowed": False,
                    "allow_aggregation": False,
                    "allow_raw_rows": False,
                    "message": f"No policy found for {user_role} accessing {data_domain}/{sensitivity_level}"
                }
            
            allowed, allow_aggregation, allow_raw_rows = result
            
            return {
                "allowed": allowed,
                "allow_aggregation": allow_aggregation,
                "allow_raw_rows": allow_raw_rows,
                "user_role": user_role,
                "data_domain": data_domain,
                "sensitivity_level": sensitivity_level,
                "message": "Access allowed" if allowed else "Access denied"
            }
            
        finally:
            conn.close()
    
    def _get_user_policies(self, user_role: str) -> Dict[str, Any]:
        """
        Get all policies for a specific user role
        
        Args:
            user_role: User's role
            
        Returns:
            Dict with all policies for the role
        """
        conn = get_db_connection(self.db_url)
        
        try:
            cur = conn.cursor()
            
            query = """
                SELECT data_domain, sensitivity_level, allowed, allow_aggregation, allow_raw_rows
                FROM retrieval_policies
                WHERE role = %s
                ORDER BY data_domain, sensitivity_level
            """
            
            cur.execute(query, (user_role,))
            results = cur.fetchall()
            
            policies = []
            for row in results:
                policies.append({
                    "data_domain": row[0],
                    "sensitivity_level": row[1],
                    "allowed": row[2],
                    "allow_aggregation": row[3],
                    "allow_raw_rows": row[4]
                })
            
            return {
                "user_role": user_role,
                "policies": policies,
                "total_policies": len(policies)
            }
            
        finally:
            conn.close()
