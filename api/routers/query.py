"""
Query Router
Handles natural language query and semantic search endpoints
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from api.models import QueryRequest, QueryResponse, SemanticSearchRequest, SearchResponse, SearchResultItem
from api.dependencies import get_database_url, get_openai_key
import pandas as pd

router = APIRouter(prefix="/api/v1", tags=["Query"])


@router.post("/query", response_model=QueryResponse)
async def query_data(
    request: QueryRequest,
    db_url: str = Depends(get_database_url),
    openai_key: str = Depends(get_openai_key)
):
    """
    Execute a natural language query against ingested data
    
    - **query**: Natural language query (e.g., "Show me revenue data for 2023")
    - **limit**: Number of relevant sheets to search (default: 3)
    
    Returns:
    - Natural language answer
    - Generated SQL query
    - Result data
    - Source citations
    """
    
    try:
        from retrieval import process_retrieval
        
        result = process_retrieval(
            user_query=request.query,
            db_url=db_url,
            openai_key=openai_key
        )
        
        # Convert DataFrame to dict if present
        data = None
        if result.get("data") is not None and isinstance(result["data"], pd.DataFrame):
            data = result["data"].to_dict(orient="records")
        
        return QueryResponse(
            success=True,
            answer=result.get("answer", "No answer generated"),
            sql=result.get("sql"),
            data=data,
            sources=result.get("sources", []),
            chart_available=result.get("chart") is not None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@router.post("/search/semantic", response_model=SearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    db_url: str = Depends(get_database_url),
    openai_key: str = Depends(get_openai_key)
):
    """
    Perform semantic search across all metadata
    
    - **query**: Search query
    - **threshold**: Similarity threshold (0.0 to 1.0, default: 0.7)
    - **limit**: Maximum results (default: 10)
    
    Returns:
    - Ranked search results with similarity scores
    """
    
    try:
        from ingest_excel import get_embedding, get_db_connection
        
        # Generate query embedding
        query_embedding = get_embedding(request.query, openai_key)
        
        # Search across sheets metadata
        conn = get_db_connection(db_url)
        
        sql = """
            SELECT 
                f.file_name,
                f.file_id::text,
                s.sheet_name,
                s.sheet_id::text,
                s.summary,
                1 - (s.summary_embedding <=> %s::vector) as similarity
            FROM sheets_metadata s
            JOIN files_metadata f ON s.file_id = f.file_id
            WHERE 1 - (s.summary_embedding <=> %s::vector) > %s
            ORDER BY similarity DESC
            LIMIT %s
        """
        
        df = pd.read_sql(
            sql, 
            conn, 
            params=(query_embedding, query_embedding, request.threshold, request.limit)
        )
        
        conn.close()
        
        # Convert to response
        results = []
        for _, row in df.iterrows():
            results.append(SearchResultItem(
                file_name=row['file_name'],
                sheet_name=row['sheet_name'],
                summary=row['summary'],
                similarity=float(row['similarity']),
                file_id=row['file_id'],
                sheet_id=row['sheet_id']
            ))
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=results,
            total_results=len(results)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")
