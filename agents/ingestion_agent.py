"""
Ingestion Agent - Handles file ingestion and metadata extraction
"""

import os
import tempfile
from typing import Dict, Any
from agents.base_agent import BaseAgent, AgentStatus
from core.config import get_config
from core.database import DatabasePool
from ingest_excel import process_excel_file, create_metadata_tables


class IngestionAgent(BaseAgent):
    """
    Ingestion Agent for processing and ingesting files.
    
    Responsibilities:
    - Process Excel/CSV files
    - Extract metadata
    - Generate embeddings
    - Store data in database
    """
    
    def __init__(self):
        super().__init__(name="IngestionAgent")
        self.config = get_config()
        self.db_pool = DatabasePool(
            self.config.database_url,
            min_conn=self.config.db_pool_min,
            max_conn=self.config.db_pool_max
        )
        
        # Ensure metadata tables exist
        with self.db_pool.get_connection() as conn:
            create_metadata_tables(conn)
        
        self.log_info("Ingestion Agent initialized")
    
    def execute(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute file ingestion.
        
        Expected payload:
        {
            "file_path": str,  # Path to file to ingest
            "file_content": bytes (optional),  # File content if not using path
            "file_name": str (optional)  # Original filename if using content
        }
        
        Returns:
            Result dictionary with success status
        """
        self._update_status(AgentStatus.PROCESSING)
        
        try:
            file_path = payload.get('file_path')
            file_content = payload.get('file_content')
            file_name = payload.get('file_name')
            
            # Handle file content (from upload)
            if file_content:
                # Create temporary file
                suffix = os.path.splitext(file_name)[1] if file_name else '.xlsx'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                    tmp_file.write(file_content)
                    file_path = tmp_file.name
                
                self.log_info(f"Created temporary file: {file_path}")
            
            if not file_path or not os.path.exists(file_path):
                raise ValueError(f"File not found: {file_path}")
            
            self.log_info(f"Processing file: {file_path}")
            
            # Process the file using existing ingestion logic
            process_excel_file(
                file_path,
                self.config.database_url,
                self.config.openrouter_api_key
            )
            
            # Clean up temporary file if created
            if file_content and os.path.exists(file_path):
                os.remove(file_path)
                self.log_info(f"Cleaned up temporary file: {file_path}")
            
            self._update_status(AgentStatus.COMPLETED)
            
            return {
                "success": True,
                "message": f"Successfully ingested file: {file_name or os.path.basename(file_path)}",
                "file_name": file_name or os.path.basename(file_path)
            }
            
        except Exception as e:
            # Clean up temporary file on error
            if file_content and file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
            
            return self.handle_error(e, {"payload": payload})
