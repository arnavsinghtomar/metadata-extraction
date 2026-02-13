"""
Test script to check agent status and functionality
"""

import os
from dotenv import load_dotenv
from agents import MasterAgent, ChartAgent, RBACAgent
import pandas as pd
import uuid
from agents.base_agent import AgentTask

# Load environment
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

print("=" * 60)
print(" AGENT STATUS CHECK")
print("=" * 60)
print()

# Initialize Master Agent
try:
    master = MasterAgent(db_url=DB_URL, openai_key=OPENROUTER_KEY)
    print(" Master Agent initialized successfully!")
    print()
except Exception as e:
    print(f" Failed to initialize Master Agent: {e}")
    exit(1)

# Get agent status
print(" Agent Status:")
print("-" * 60)

status = master.get_agent_status()
for agent_name, info in status.items():
    print(f"\n {agent_name.upper()}")
    print(f"   Description: {info['description']}")
    print(f"   Status: {' Enabled' if info['enabled'] else '❌ Disabled'}")
    print(f"   Capabilities: {info['capabilities']}")

print()
print("=" * 60)
print("TESTING AGENTS")
print("=" * 60)
print()

# Test 1: Maintenance Agent - Validate Database
print("Test 1: Database Validation")
print("-" * 60)
try:
    task = master.create_task(
        task_type="validate",
        payload={}
    )
    response = master.execute(task)
    
    if response.status == "completed":
        print(f"[OK] Status: {response.status}")
        print(f"Files: {response.result['files_count']}")
        print(f"Sheets: {response.result['sheets_count']}")
        print(f"Healthy: {response.result['is_healthy']}")
    else:
        print(f"[ERROR] Failed: {response.error}")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 2: ChartAgent - Verify Initialization
print("Test 2: ChartAgent Initialization")
print("-" * 60)
try:
    chart_agent = ChartAgent(openai_key=OPENROUTER_KEY)
    print(f"[OK] ChartAgent initialized: {chart_agent.name}")
    print(f"   Capabilities: {len(chart_agent.capabilities)}")
    for cap in chart_agent.capabilities:
        print(f"   - {cap.name}: {cap.description}")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 3: RBACAgent - Verify Initialization
print("Test 3: RBACAgent Initialization")
print("-" * 60)
try:
    rbac_agent = RBACAgent(db_url=DB_URL)
    print(f"[OK] RBACAgent initialized: {rbac_agent.name}")
    print(f"   Capabilities: {len(rbac_agent.capabilities)}")
    for cap in rbac_agent.capabilities:
        print(f"   - {cap.name}: {cap.description}")
except Exception as e:
    print(f"❌ Error: {e}")

print()

# Test 4: RBACAgent - SQL Validation
print("Test 4: RBACAgent SQL Validation")
print("-" * 60)
try:
    rbac_agent = RBACAgent(db_url=DB_URL)
    
    # Test with aggregation (should pass)
    task = AgentTask(
        task_id=str(uuid.uuid4()),
        task_type="validate_sql",
        payload={
            "sql_query": "SELECT vendor, SUM(amount) FROM table GROUP BY vendor",
            "sheet_info": {"allow_raw_rows": False, "data_domain": "finance"}
        }
    )
    response = rbac_agent.execute(task)
    print(f"[OK] Aggregation query validation: {response.status.value}")
    
    # Test without aggregation (should fail)
    task2 = AgentTask(
        task_id=str(uuid.uuid4()),
        task_type="validate_sql",
        payload={
            "sql_query": "SELECT * FROM table",
            "sheet_info": {"allow_raw_rows": False, "data_domain": "finance"}
        }
    )
    response2 = rbac_agent.execute(task2)
    print(f"[OK] Raw query blocked: {response2.status.value == 'failed'}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 60)
print("AGENT SYSTEM READY!")
print("=" * 60)
print()
print("Available commands:")
print("  - master.create_task('ingest', {...})  # Upload files")
print("  - master.create_task('query', {...})   # Ask questions")
print("  - master.create_task('analyze', {...}) # Business health")
print("  - master.create_task('cleanup', {...}) # Database cleanup")
print()
print("New Agents:")
print("  - ChartAgent: Generates data visualizations")
print("  - RBACAgent: Manages role-based access control")
print()
