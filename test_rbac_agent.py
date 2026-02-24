"""
Unit tests for RBACAgent
"""

import pytest
from agents.rbac_agent import RBACAgent
from agents.base_agent import AgentTask, TaskStatus
import uuid
import os


class TestRBACAgent:
    """Test suite for RBACAgent"""
    
    @pytest.fixture
    def rbac_agent(self):
        """Create an RBACAgent instance for testing"""
        # Use environment variable or test database URL
        db_url = os.getenv("DATABASE_URL", "postgresql://test:test@localhost/test")
        return RBACAgent(db_url=db_url)
    
    def test_agent_initialization(self, rbac_agent):
        """Test that RBACAgent initializes correctly"""
        assert rbac_agent.name == "RBACAgent"
        assert rbac_agent.enabled is True
        assert len(rbac_agent.capabilities) == 4
    
    def test_validate_task_validate_sql(self, rbac_agent):
        """Test task validation for validate_sql task type"""
        valid_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="validate_sql",
            payload={
                "sql_query": "SELECT * FROM table",
                "sheet_info": {"allow_raw_rows": True}
            }
        )
        assert rbac_agent.validate_task(valid_task) is True
        
        invalid_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="validate_sql",
            payload={"sql_query": "SELECT * FROM table"}  # Missing sheet_info
        )
        assert rbac_agent.validate_task(invalid_task) is False
    
    def test_validate_task_check_permission(self, rbac_agent):
        """Test task validation for check_permission task type"""
        valid_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="check_permission",
            payload={
                "user_role": "Analyst",
                "data_domain": "finance",
                "sensitivity_level": "confidential"
            }
        )
        assert rbac_agent.validate_task(valid_task) is True
    
    def test_validate_sql_access_allowed(self, rbac_agent):
        """Test SQL validation when raw access is allowed"""
        sheet_info = {
            "allow_raw_rows": True,
            "data_domain": "finance"
        }
        
        result = rbac_agent._validate_sql_access("SELECT * FROM table", sheet_info)
        assert result["valid"] is True
        assert result["access_level"] == "raw"
    
    def test_validate_sql_access_aggregation_required(self, rbac_agent):
        """Test SQL validation when only aggregation is allowed"""
        sheet_info = {
            "allow_raw_rows": False,
            "data_domain": "finance"
        }
        
        # Valid aggregation query
        result = rbac_agent._validate_sql_access(
            "SELECT vendor, SUM(amount) FROM table GROUP BY vendor",
            sheet_info
        )
        assert result["valid"] is True
        assert result["access_level"] == "aggregated"
    
    def test_validate_sql_access_denied_select_star(self, rbac_agent):
        """Test SQL validation denies SELECT * when raw access not allowed"""
        sheet_info = {
            "allow_raw_rows": False,
            "data_domain": "finance"
        }
        
        with pytest.raises(PermissionError) as exc_info:
            rbac_agent._validate_sql_access("SELECT * FROM table", sheet_info)
        
        assert "Access Denied" in str(exc_info.value)
    
    def test_validate_sql_access_denied_no_aggregation(self, rbac_agent):
        """Test SQL validation denies queries without aggregation"""
        sheet_info = {
            "allow_raw_rows": False,
            "data_domain": "finance"
        }
        
        with pytest.raises(PermissionError) as exc_info:
            rbac_agent._validate_sql_access("SELECT vendor, amount FROM table", sheet_info)
        
        assert "aggregation functions" in str(exc_info.value)
    
    def test_execute_validate_sql_task_success(self, rbac_agent):
        """Test executing a validate_sql task successfully"""
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="validate_sql",
            payload={
                "sql_query": "SELECT COUNT(*) FROM table",
                "sheet_info": {"allow_raw_rows": False, "data_domain": "finance"}
            }
        )
        
        response = rbac_agent.execute(task)
        assert response.status == TaskStatus.COMPLETED
        assert response.result["valid"] is True
    
    def test_execute_validate_sql_task_failure(self, rbac_agent):
        """Test executing a validate_sql task that fails"""
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="validate_sql",
            payload={
                "sql_query": "SELECT * FROM table",
                "sheet_info": {"allow_raw_rows": False, "data_domain": "finance"}
            }
        )
        
        response = rbac_agent.execute(task)
        assert response.status == TaskStatus.FAILED
        assert "Access Denied" in response.error
    
    def test_execute_invalid_task(self, rbac_agent):
        """Test executing an invalid task"""
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="invalid_type",
            payload={}
        )
        
        response = rbac_agent.execute(task)
        assert response.status == TaskStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
