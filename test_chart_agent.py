"""
Unit tests for ChartAgent
"""

import pytest
import pandas as pd
from agents.chart_agent import ChartAgent
from agents.base_agent import AgentTask, TaskStatus
import uuid


class TestChartAgent:
    """Test suite for ChartAgent"""
    
    @pytest.fixture
    def chart_agent(self):
        """Create a ChartAgent instance for testing"""
        # Use a dummy API key for testing
        return ChartAgent(openai_key="test_key")
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample dataframe for testing"""
        return pd.DataFrame({
            'category': ['A', 'B', 'C', 'D'],
            'value': [10, 20, 30, 40],
            'date': pd.date_range('2024-01-01', periods=4)
        })
    
    def test_agent_initialization(self, chart_agent):
        """Test that ChartAgent initializes correctly"""
        assert chart_agent.name == "ChartAgent"
        assert chart_agent.enabled is True
        assert len(chart_agent.capabilities) == 3
    
    def test_validate_task_decide_chart(self, chart_agent):
        """Test task validation for decide_chart task type"""
        valid_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="decide_chart",
            payload={
                "user_query": "Show me the data",
                "dataframe": pd.DataFrame()
            }
        )
        assert chart_agent.validate_task(valid_task) is True
        
        invalid_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="decide_chart",
            payload={"user_query": "Show me the data"}  # Missing dataframe
        )
        assert chart_agent.validate_task(invalid_task) is False
    
    def test_validate_task_render_chart(self, chart_agent):
        """Test task validation for render_chart task type"""
        valid_task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="render_chart",
            payload={
                "dataframe": pd.DataFrame(),
                "chart_spec": {"show_chart": True, "chart_type": "bar"}
            }
        )
        assert chart_agent.validate_task(valid_task) is True
    
    def test_validate_chart_spec(self, chart_agent, sample_dataframe):
        """Test chart specification validation"""
        valid_spec = {
            "show_chart": True,
            "chart_type": "bar",
            "x_axis": "category",
            "y_axis": "value"
        }
        assert chart_agent._validate_chart_spec(sample_dataframe, valid_spec) is True
        
        invalid_spec = {
            "show_chart": True,
            "chart_type": "bar",
            "x_axis": "nonexistent_column",
            "y_axis": "value"
        }
        assert chart_agent._validate_chart_spec(sample_dataframe, invalid_spec) is False
    
    def test_render_chart_bar(self, chart_agent, sample_dataframe):
        """Test rendering a bar chart"""
        chart_spec = {
            "show_chart": True,
            "chart_type": "bar",
            "x_axis": "category",
            "y_axis": "value",
            "reason": "Test bar chart"
        }
        
        fig = chart_agent._render_chart(sample_dataframe, chart_spec)
        assert fig is not None
        assert fig.layout.title.text == "Test bar chart"
    
    def test_render_chart_line(self, chart_agent, sample_dataframe):
        """Test rendering a line chart"""
        chart_spec = {
            "show_chart": True,
            "chart_type": "line",
            "x_axis": "date",
            "y_axis": "value",
            "reason": "Test line chart"
        }
        
        fig = chart_agent._render_chart(sample_dataframe, chart_spec)
        assert fig is not None
    
    def test_render_chart_no_show(self, chart_agent, sample_dataframe):
        """Test that no chart is rendered when show_chart is False"""
        chart_spec = {
            "show_chart": False
        }
        
        fig = chart_agent._render_chart(sample_dataframe, chart_spec)
        assert fig is None
    
    def test_execute_render_chart_task(self, chart_agent, sample_dataframe):
        """Test executing a render_chart task"""
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="render_chart",
            payload={
                "dataframe": sample_dataframe,
                "chart_spec": {
                    "show_chart": True,
                    "chart_type": "bar",
                    "x_axis": "category",
                    "y_axis": "value"
                }
            }
        )
        
        response = chart_agent.execute(task)
        assert response.status == TaskStatus.COMPLETED
        assert response.result is not None
    
    def test_execute_invalid_task(self, chart_agent):
        """Test executing an invalid task"""
        task = AgentTask(
            task_id=str(uuid.uuid4()),
            task_type="invalid_type",
            payload={}
        )
        
        response = chart_agent.execute(task)
        assert response.status == TaskStatus.FAILED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
