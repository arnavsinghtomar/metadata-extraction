"""
Database connection pooling for efficient resource management
"""

import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class DatabasePool:
    """
    PostgreSQL connection pool manager.
    
    Provides efficient connection pooling to avoid creating
    new connections for every database operation.
    """
    
    _instance: Optional['DatabasePool'] = None
    _pool: Optional[pool.SimpleConnectionPool] = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one pool exists"""
        if cls._instance is None:
            cls._instance = super(DatabasePool, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, db_url: str, min_conn: int = 1, max_conn: int = 10):
        """
        Initialize connection pool.
        
        Args:
            db_url: PostgreSQL connection URL
            min_conn: Minimum number of connections in pool
            max_conn: Maximum number of connections in pool
        """
        if self._pool is None:
            try:
                self._pool = pool.SimpleConnectionPool(
                    min_conn,
                    max_conn,
                    db_url
                )
                logger.info(f"Database pool created: {min_conn}-{max_conn} connections")
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}")
                raise
    
    @contextmanager
    def get_connection(self):
        """
        Context manager to get a connection from the pool.
        
        Usage:
            with db_pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM table")
        
        Yields:
            psycopg2.connection: Database connection
        """
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)
    
    @contextmanager
    def get_cursor(self, commit: bool = False):
        """
        Context manager to get a cursor with automatic connection handling.
        
        Args:
            commit: Whether to commit after cursor operations
            
        Usage:
            with db_pool.get_cursor(commit=True) as cur:
                cur.execute("INSERT INTO table VALUES (%s)", (value,))
        
        Yields:
            psycopg2.cursor: Database cursor
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                if commit:
                    conn.commit()
            except Exception as e:
                conn.rollback()
                raise
            finally:
                cursor.close()
    
    def close_all(self):
        """Close all connections in the pool"""
        if self._pool:
            self._pool.closeall()
            logger.info("All database connections closed")
    
    def get_stats(self) -> dict:
        """
        Get pool statistics.
        
        Returns:
            Dictionary with pool stats
        """
        if self._pool:
            return {
                "min_connections": self._pool.minconn,
                "max_connections": self._pool.maxconn,
                "available": len(self._pool._pool),
                "in_use": len(self._pool._used)
            }
        return {}


# Convenience function for getting a connection
def get_db_connection(db_url: str):
    """
    Get a database connection (legacy compatibility).
    
    Args:
        db_url: PostgreSQL connection URL
        
    Returns:
        psycopg2.connection
    """
    try:
        return psycopg2.connect(db_url)
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise
