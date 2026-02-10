"""
File Ingestion Router
Handles file upload and processing endpoints
"""

import os
import uuid
import tempfile
import time
from typing import List
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from api.models import FileUploadResponse, BatchUploadResponse, ErrorResponse
from api.dependencies import get_ingestion_agent, validate_file_extension, get_file_extension
from agents.ingestion_agent import IngestionAgent
from agents.base_agent import AgentTask, TaskStatus

router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.xlsx', '.xls', '.pdf', '.csv'}


def process_file_task(file_path: str, original_filename: str, agent: IngestionAgent) -> dict:
    """
    Background task to process a file
    """
    task = AgentTask(
        task_id=str(uuid.uuid4()),
        task_type="ingest",
        payload={
            "file_path": file_path,
            "original_filename": original_filename
        }
    )
    
    response = agent.execute(task)
    
    # Cleanup temp file
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except:
        pass
    
    return {
        "task_id": response.task_id,
        "status": response.status,
        "result": response.result,
        "error": response.error,
        "execution_time": response.execution_time
    }


@router.post("/file", response_model=FileUploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    agent: IngestionAgent = Depends(get_ingestion_agent)
):
    """
    Upload and process a single file (Excel, PDF, or CSV)
    
    - **file**: File to upload (max 50MB)
    
    Returns:
    - File metadata and processing status
    """
    
    # Validate file extension
    if not validate_file_extension(file.filename, ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Read file content
    try:
        content = await file.read()
        
        # Check file size
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )
        
        # Save to temporary file
        file_ext = get_file_extension(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # Process file
        start_time = time.time()
        result = process_file_task(tmp_path, file.filename, agent)
        
        # Prepare response
        if result["status"] == TaskStatus.COMPLETED:
            return FileUploadResponse(
                success=True,
                task_id=result["task_id"],
                file_name=file.filename,
                status=result["result"].get("status", "SUCCESS"),
                message=result["result"].get("message", "File processed successfully"),
                execution_time=result["execution_time"]
            )
        else:
            return FileUploadResponse(
                success=False,
                task_id=result["task_id"],
                file_name=file.filename,
                status="FAILED",
                message=result.get("error", "Processing failed"),
                execution_time=result["execution_time"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.post("/batch", response_model=BatchUploadResponse)
async def upload_batch(
    files: List[UploadFile] = File(...),
    agent: IngestionAgent = Depends(get_ingestion_agent)
):
    """
    Upload and process multiple files concurrently
    
    - **files**: List of files to upload (max 20 files)
    
    Returns:
    - Batch processing results
    """
    
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed per batch")
    
    results = []
    successful = 0
    failed = 0
    
    import concurrent.futures
    
    def process_single_file(upload_file: UploadFile):
        """Process a single file in the batch"""
        try:
            # Validate
            if not validate_file_extension(upload_file.filename, ALLOWED_EXTENSIONS):
                return FileUploadResponse(
                    success=False,
                    task_id=str(uuid.uuid4()),
                    file_name=upload_file.filename,
                    status="FAILED",
                    message=f"Invalid file type"
                )
            
            # Read and save
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            content = loop.run_until_complete(upload_file.read())
            
            if len(content) > MAX_FILE_SIZE:
                return FileUploadResponse(
                    success=False,
                    task_id=str(uuid.uuid4()),
                    file_name=upload_file.filename,
                    status="FAILED",
                    message="File too large"
                )
            
            file_ext = get_file_extension(upload_file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Process
            result = process_file_task(tmp_path, upload_file.filename, agent)
            
            if result["status"] == TaskStatus.COMPLETED:
                return FileUploadResponse(
                    success=True,
                    task_id=result["task_id"],
                    file_name=upload_file.filename,
                    status=result["result"].get("status", "SUCCESS"),
                    message=result["result"].get("message", "Success"),
                    execution_time=result["execution_time"]
                )
            else:
                return FileUploadResponse(
                    success=False,
                    task_id=result["task_id"],
                    file_name=upload_file.filename,
                    status="FAILED",
                    message=result.get("error", "Failed"),
                    execution_time=result["execution_time"]
                )
                
        except Exception as e:
            return FileUploadResponse(
                success=False,
                task_id=str(uuid.uuid4()),
                file_name=upload_file.filename,
                status="FAILED",
                message=str(e)
            )
    
    # Process files concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(files), 8)) as executor:
        futures = [executor.submit(process_single_file, f) for f in files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if result.success:
                successful += 1
            else:
                failed += 1
    
    return BatchUploadResponse(
        success=True,
        total_files=len(files),
        successful=successful,
        failed=failed,
        results=results
    )
