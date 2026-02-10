"""
Metadata Retrieval Router
Handles metadata retrieval and management endpoints
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import pandas as pd
from uuid import UUID

from api.models import (
    FileMetadataResponse, 
    SheetMetadataResponse, 
    FileDetailResponse,
    StatsResponse
)
from api.dependencies import get_db_connection

router = APIRouter(prefix="/api/v1", tags=["Metadata"])


@router.get("/files", response_model=List[FileMetadataResponse])
async def list_files(
    limit: Optional[int] = 100,
    offset: Optional[int] = 0,
    conn=Depends(get_db_connection)
):
    """
    List all ingested files with metadata
    
    - **limit**: Maximum number of files to return (default: 100)
    - **offset**: Number of files to skip (default: 0)
    
    Returns:
    - List of file metadata
    """
    
    try:
        query = """
            SELECT 
                file_id::text,
                file_name,
                uploaded_at,
                num_sheets,
                num_pages,
                summary,
                keywords,
                file_hash
            FROM files_metadata
            ORDER BY uploaded_at DESC
            LIMIT %s OFFSET %s
        """
        
        df = pd.read_sql(query, conn, params=(limit, offset))
        
        if df.empty:
            return []
        
        # Convert to response models
        files = []
        for _, row in df.iterrows():
            files.append(FileMetadataResponse(
                file_id=row['file_id'],
                file_name=row['file_name'],
                uploaded_at=row['uploaded_at'],
                num_sheets=row.get('num_sheets'),
                num_pages=row.get('num_pages'),
                summary=row.get('summary'),
                keywords=row.get('keywords'),
                file_hash=row.get('file_hash')
            ))
        
        return files
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving files: {str(e)}")


@router.get("/files/{file_id}", response_model=FileDetailResponse)
async def get_file_detail(
    file_id: str,
    conn=Depends(get_db_connection)
):
    """
    Get detailed metadata for a specific file including all sheets
    
    - **file_id**: UUID of the file
    
    Returns:
    - Detailed file metadata with sheets
    """
    
    try:
        # Get file metadata
        file_query = """
            SELECT 
                file_id::text,
                file_name,
                uploaded_at,
                num_sheets,
                num_pages,
                summary,
                keywords,
                file_hash
            FROM files_metadata
            WHERE file_id = %s
        """
        
        file_df = pd.read_sql(file_query, conn, params=(file_id,))
        
        if file_df.empty:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_row = file_df.iloc[0]
        file_metadata = FileMetadataResponse(
            file_id=file_row['file_id'],
            file_name=file_row['file_name'],
            uploaded_at=file_row['uploaded_at'],
            num_sheets=file_row.get('num_sheets'),
            num_pages=file_row.get('num_pages'),
            summary=file_row.get('summary'),
            keywords=file_row.get('keywords'),
            file_hash=file_row.get('file_hash')
        )
        
        # Get sheets metadata
        sheets_query = """
            SELECT 
                sheet_id::text,
                file_id::text,
                sheet_name,
                table_name,
                num_rows,
                num_columns,
                category,
                summary,
                keywords,
                columns_metadata
            FROM sheets_metadata
            WHERE file_id = %s
            ORDER BY sheet_name
        """
        
        sheets_df = pd.read_sql(sheets_query, conn, params=(file_id,))
        
        sheets = []
        for _, row in sheets_df.iterrows():
            sheets.append(SheetMetadataResponse(
                sheet_id=row['sheet_id'],
                file_id=row['file_id'],
                sheet_name=row['sheet_name'],
                table_name=row['table_name'],
                num_rows=row['num_rows'],
                num_columns=row['num_columns'],
                category=row.get('category'),
                summary=row.get('summary'),
                keywords=row.get('keywords'),
                columns_metadata=row.get('columns_metadata')
            ))
        
        return FileDetailResponse(
            file=file_metadata,
            sheets=sheets
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving file details: {str(e)}")


@router.get("/sheets/{sheet_id}", response_model=SheetMetadataResponse)
async def get_sheet_metadata(
    sheet_id: str,
    conn=Depends(get_db_connection)
):
    """
    Get metadata for a specific sheet
    
    - **sheet_id**: UUID of the sheet
    
    Returns:
    - Sheet metadata
    """
    
    try:
        query = """
            SELECT 
                sheet_id::text,
                file_id::text,
                sheet_name,
                table_name,
                num_rows,
                num_columns,
                category,
                summary,
                keywords,
                columns_metadata
            FROM sheets_metadata
            WHERE sheet_id = %s
        """
        
        df = pd.read_sql(query, conn, params=(sheet_id,))
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Sheet not found")
        
        row = df.iloc[0]
        
        return SheetMetadataResponse(
            sheet_id=row['sheet_id'],
            file_id=row['file_id'],
            sheet_name=row['sheet_name'],
            table_name=row['table_name'],
            num_rows=row['num_rows'],
            num_columns=row['num_columns'],
            category=row.get('category'),
            summary=row.get('summary'),
            keywords=row.get('keywords'),
            columns_metadata=row.get('columns_metadata')
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving sheet metadata: {str(e)}")


@router.delete("/files/{file_id}")
async def delete_file(
    file_id: str,
    conn=Depends(get_db_connection)
):
    """
    Delete a file and all associated data
    
    - **file_id**: UUID of the file to delete
    
    Returns:
    - Success message
    """
    
    try:
        cursor = conn.cursor()
        
        # Check if file exists
        cursor.execute("SELECT file_name FROM files_metadata WHERE file_id = %s", (file_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_name = result[0]
        
        # Delete file (cascade will delete sheets and chunks)
        cursor.execute("DELETE FROM files_metadata WHERE file_id = %s", (file_id,))
        conn.commit()
        
        return JSONResponse({
            "success": True,
            "message": f"File '{file_name}' and all associated data deleted successfully"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@router.get("/stats", response_model=StatsResponse)
async def get_stats(conn=Depends(get_db_connection)):
    """
    Get system statistics
    
    Returns:
    - Total files, sheets, rows, and storage info
    """
    
    try:
        cursor = conn.cursor()
        
        # Total files
        cursor.execute("SELECT COUNT(*) FROM files_metadata")
        total_files = cursor.fetchone()[0]
        
        # Total sheets
        cursor.execute("SELECT COUNT(*) FROM sheets_metadata")
        total_sheets = cursor.fetchone()[0]
        
        # Total rows
        cursor.execute("SELECT SUM(num_rows) FROM sheets_metadata")
        total_rows_result = cursor.fetchone()[0]
        total_rows = int(total_rows_result) if total_rows_result else 0
        
        # Total PDF chunks
        cursor.execute("SELECT COUNT(*) FROM pdf_chunks")
        total_pdf_chunks = cursor.fetchone()[0]
        
        return StatsResponse(
            total_files=total_files,
            total_sheets=total_sheets,
            total_rows=total_rows,
            total_pdf_chunks=total_pdf_chunks
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving stats: {str(e)}")
