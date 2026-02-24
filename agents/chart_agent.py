"""
Chart Agent - Handles visualization generation and rendering
"""

import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import Field
import pandas as pd
import plotly.express as px
from openai import OpenAI

from .base_agent import BaseAgent, AgentTask, AgentResponse, TaskStatus, AgentCapability

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartAgent(BaseAgent):
    """
    Agent responsible for chart generation and visualization
    Analyzes data and creates appropriate Plotly visualizations
    """
    
    name: str = Field(default="ChartAgent")
    description: str = Field(
        default="Generates data visualizations using Plotly based on query results and data analysis"
    )
    capabilities: list = Field(
        default_factory=lambda: [
            AgentCapability(
                name="chart_decision",
                description="Analyze data and decide appropriate chart type",
                required_tools=["openai", "pandas"]
            ),
            AgentCapability(
                name="chart_rendering",
                description="Render Plotly charts (bar, line, pie, scatter)",
                required_tools=["plotly"]
            ),
            AgentCapability(
                name="chart_validation",
                description="Validate chart specifications against dataframe",
                required_tools=["pandas"]
            )
        ]
    )
    
    openai_key: Optional[str] = Field(default=None)
    
    def __init__(self, openai_key: str, **data):
        super().__init__(**data)
        self.openai_key = openai_key
    
    def validate_task(self, task: AgentTask) -> bool:
        """Validate chart generation task"""
        if task.task_type not in ["decide_chart", "render_chart", "generate_visualization"]:
            return False
        
        # For decide_chart and generate_visualization, need query and dataframe
        if task.task_type in ["decide_chart", "generate_visualization"]:
            required_fields = ["user_query", "dataframe"]
            return all(field in task.payload for field in required_fields)
        
        # For render_chart, need dataframe and chart_spec
        if task.task_type == "render_chart":
            required_fields = ["dataframe", "chart_spec"]
            return all(field in task.payload for field in required_fields)
        
        return True
    
    def execute(self, task: AgentTask) -> AgentResponse:
        """
        Execute chart generation task
        
        Expected task.payload:
            For decide_chart:
                - user_query: str (user's question)
                - dataframe: pd.DataFrame (query results)
            
            For render_chart:
                - dataframe: pd.DataFrame (data to visualize)
                - chart_spec: dict (chart specification)
            
            For generate_visualization:
                - user_query: str (user's question)
                - dataframe: pd.DataFrame (query results)
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
            if task.task_type == "decide_chart":
                result = self._decide_chart(
                    task.payload["user_query"],
                    task.payload["dataframe"]
                )
            elif task.task_type == "render_chart":
                result = self._render_chart(
                    task.payload["dataframe"],
                    task.payload["chart_spec"]
                )
            elif task.task_type == "generate_visualization":
                # Combined operation: decide + render
                df = task.payload["dataframe"]
                query = task.payload["user_query"]
                
                chart_spec = self._decide_chart(query, df)
                
                if chart_spec.get("show_chart"):
                    chart = self._render_chart(df, chart_spec)
                    result = {
                        "chart_spec": chart_spec,
                        "chart": chart,
                        "success": chart is not None
                    }
                else:
                    result = {
                        "chart_spec": chart_spec,
                        "chart": None,
                        "success": False,
                        "reason": chart_spec.get("reason", "Chart not appropriate for this data")
                    }
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
            
        except Exception as e:
            return self.handle_error(task, e)
    
    def _decide_chart(self, user_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Decide if a visualization is useful and return chart specification
        
        Args:
            user_query: User's question
            df: Query results dataframe
            
        Returns:
            Chart specification dict with show_chart, chart_type, x_axis, y_axis, reason
        """
        logger.info("Deciding chart type for query")
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.openai_key
        )
        
        columns_info = ", ".join(
            [f"{c} ({df[c].dtype})" for c in df.columns]
        )
        
        prompt = f"""
You are an expert data analyst deciding whether to show a chart.

User Question:
"{user_query}"

Available Data Columns (ONLY these can be used):
{columns_info}

IMPORTANT DEFINITIONS:
- Categorical columns: object, string, text
- Numeric columns: int, float, double
- Time columns: any column whose name contains date, time, year, month, day
  (even if dtype is object/string)

CRITICAL RULES:
1. You MUST choose x_axis and y_axis ONLY from the listed columns.
2. DO NOT invent columns.
3. If data is wide (multiple numeric columns), choose ONE numeric column and proceed.
4. If table has ≤2 rows, show a chart if comparison is still meaningful.
5. Line charts are allowed if a time-like column exists (name-based).
6. Pie charts ONLY if categories ≤6.
7. You MUST make a best-effort decision.
   Return show_chart=false ONLY if chart is clearly impossible.

Decision logic:
- Category + numeric → bar
- Time + numeric → line
- Single numeric only → no chart
- No category or time column → no chart

Respond in STRICT JSON ONLY:

{{
  "show_chart": true,
  "chart_type": "bar",
  "x_axis": "column_name",
  "y_axis": "column_name",
  "reason": "short justification"
}}
"""
        
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON if it's wrapped in text or markdown
            if "{" in content and "}" in content:
                content = content[content.find("{"):content.rfind("}")+1]
            
            parsed_json = json.loads(content)
            logger.info(f"Chart decision: {parsed_json}")
            return parsed_json
            
        except Exception as e:
            logger.error(f"Failed to decide chart: {e}")
            return {"show_chart": False, "reason": f"Error: {str(e)}"}
    
    def _render_chart(self, df: pd.DataFrame, chart_spec: Dict[str, Any]):
        """
        Render a Plotly chart based on the specification
        
        Args:
            df: Data to visualize
            chart_spec: Chart specification from _decide_chart
            
        Returns:
            Plotly figure object or None
        """
        logger.info("Rendering chart")
        
        if not chart_spec.get("show_chart"):
            return None
        
        chart_type = chart_spec.get("chart_type")
        x = chart_spec.get("x_axis")
        y = chart_spec.get("y_axis")
        
        # Validate columns exist in DF
        if not self._validate_chart_spec(df, chart_spec):
            logger.error("Chart validation failed")
            return None
        
        try:
            title = chart_spec.get("reason", "Data Visualization")
            
            if chart_type == "bar":
                return px.bar(df, x=x, y=y, title=title)
            elif chart_type == "line":
                return px.line(df, x=x, y=y, title=title)
            elif chart_type == "pie":
                return px.pie(df, names=x, values=y, title=title)
            elif chart_type == "scatter":
                return px.scatter(df, x=x, y=y, title=title)
            else:
                logger.warning(f"Unknown chart type: {chart_type}")
                return None
                
        except Exception as e:
            logger.error(f"Chart rendering failed: {e}")
            return None
    
    def _validate_chart_spec(self, df: pd.DataFrame, chart_spec: Dict[str, Any]) -> bool:
        """
        Validate that chart specification matches dataframe columns
        
        Args:
            df: Dataframe to validate against
            chart_spec: Chart specification
            
        Returns:
            True if valid, False otherwise
        """
        x = chart_spec.get("x_axis")
        y = chart_spec.get("y_axis")
        
        if x and x not in df.columns:
            logger.error(f"Chart validation failed: {x} not in {df.columns.tolist()}")
            return False
        
        if y and y not in df.columns:
            logger.error(f"Chart validation failed: {y} not in {df.columns.tolist()}")
            return False
        
        return True
