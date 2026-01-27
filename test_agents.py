"""
Test script to check agent status and functionality
"""

import os
from dotenv import load_dotenv
from agents import MasterAgent

# Load environment
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

print("=" * 60)
print("🤖 AGENT STATUS CHECK")
print("=" * 60)
print()

# Initialize Master Agent
try:
    master = MasterAgent(db_url=DB_URL, openai_key=OPENROUTER_KEY)
    print("✅ Master Agent initialized successfully!")
    print()
except Exception as e:
    print(f"❌ Failed to initialize Master Agent: {e}")
    exit(1)

# Get agent status
print("📊 Agent Status:")
print("-" * 60)

status = master.get_agent_status()
for agent_name, info in status.items():
    print(f"\n🤖 {agent_name.upper()}")
    print(f"   Description: {info['description']}")
    print(f"   Status: {'✅ Enabled' if info['enabled'] else '❌ Disabled'}")
    print(f"   Capabilities: {info['capabilities']}")

print()
print("=" * 60)
print("🔧 TESTING AGENTS")
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
        print(f"✅ Status: {response.status}")
        print(f"📊 Files: {response.result['files_count']}")
        print(f"📊 Sheets: {response.result['sheets_count']}")
        print(f"💚 Healthy: {response.result['is_healthy']}")
    else:
        print(f"❌ Failed: {response.error}")
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 60)
print("✨ AGENT SYSTEM READY!")
print("=" * 60)
print()
print("Available commands:")
print("  - master.create_task('ingest', {...})  # Upload files")
print("  - master.create_task('query', {...})   # Ask questions")
print("  - master.create_task('analyze', {...}) # Business health")
print("  - master.create_task('cleanup', {...}) # Database cleanup")
print()
