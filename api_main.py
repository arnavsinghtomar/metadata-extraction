"""
FastAPI Main Application
Metadata Ingestion REST API
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from api.routers import ingestion, metadata, query
from api.models import HealthResponse
from api import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Metadata Ingestion API",
    description="REST API for file ingestion, metadata extraction, and intelligent querying",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Include routers
app.include_router(ingestion.router)
app.include_router(metadata.router)
app.include_router(query.router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Metadata Ingestion API",
        "version": __version__,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc"
    }


# Health check endpoint
@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
    - API status and database connectivity
    """
    
    try:
        from api.dependencies import get_database_url
        from ingest_excel import get_db_connection
        
        db_url = get_database_url()
        
        # Test database connection
        db_connected = False
        try:
            conn = get_db_connection(db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            db_connected = True
        except:
            db_connected = False
        
        return HealthResponse(
            status="healthy" if db_connected else "degraded",
            version=__version__,
            database_connected=db_connected,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            version=__version__,
            database_connected=False,
            timestamp=datetime.now()
        )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info(f"Starting Metadata Ingestion API v{__version__}")
    logger.info("API documentation available at /docs")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("Shutting down Metadata Ingestion API")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
