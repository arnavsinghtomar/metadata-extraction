"""
Test script to verify ingestion agent works correctly
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.master_agent import get_master_agent
from core.schemas import TaskType

def test_ingestion():
    """Test file ingestion through master agent"""
    
    load_dotenv()
    
    # Get master agent
    master = get_master_agent()
    
    print("=" * 60)
    print("Testing Ingestion Agent")
    print("=" * 60)
    
    # Test with a file path (you'll need to provide actual file)
    test_file = input("Enter path to Excel file to test (or press Enter to skip): ").strip()
    
    if not test_file:
        print("\nNo file provided. Skipping ingestion test.")
        print("\nTo test ingestion, run:")
        print("  python test_ingestion.py")
        print("  Then provide path to an Excel file")
        return
    
    if not os.path.exists(test_file):
        print(f"\nError: File not found: {test_file}")
        return
    
    print(f"\nIngesting file: {test_file}")
    print("-" * 60)
    
    # Create ingestion task
    result = master.route_task(
        task_type=TaskType.INGESTION,
        payload={
            "file_path": test_file
        }
    )
    
    print("\nResult:")
    print(f"  Success: {result.get('success')}")
    print(f"  Message: {result.get('message', result.get('error', 'N/A'))}")
    
    if result.get('success'):
        print("\n✅ Ingestion successful!")
        print("\nYou can now:")
        print("  1. Check Neon database to see the ingested data")
        print("  2. Run the Streamlit app to view the data")
        print("  3. Query the data using the query agent")
    else:
        print("\n❌ Ingestion failed!")
        print(f"\nError details: {result.get('error')}")
    
    print("=" * 60)

if __name__ == "__main__":
    test_ingestion()
