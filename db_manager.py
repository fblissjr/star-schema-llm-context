"""
DuckDB connection manager with proper transaction handling and connection pooling
"""

import duckdb
import threading
from contextlib import contextmanager
from typing import Optional, Any, List, Dict
import logging

logger = logging.getLogger(__name__)


class DuckDBManager:
    """Thread-safe DuckDB connection manager with transaction support"""
    
    def __init__(self, db_path: str = "data/llm_memory.duckdb"):
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get thread-local connection"""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = duckdb.connect(self.db_path)
            # Set performance settings
            self._local.conn.execute("SET threads = 4")
            self._local.conn.execute("SET memory_limit = '4GB'")
        return self._local.conn
    
    @contextmanager
    def transaction(self):
        """Context manager for transactions"""
        conn = self._get_connection()
        try:
            conn.begin()
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction rolled back: {e}")
            raise
    
    def execute(self, query: str, params: Optional[List[Any]] = None) -> duckdb.DuckDBPyRelation:
        """Execute a query with automatic transaction handling for DML"""
        conn = self._get_connection()
        
        # Determine if this is a DML statement
        query_upper = query.strip().upper()
        is_dml = any(query_upper.startswith(cmd) for cmd in ['INSERT', 'UPDATE', 'DELETE', 'MERGE'])
        
        try:
            if is_dml:
                with self.transaction():
                    return conn.execute(query, params or [])
            else:
                return conn.execute(query, params or [])
        except duckdb.CatalogException as e:
            # Don't log catalog errors during schema creation
            if 'already exists' not in str(e).lower() and 'does not exist' not in str(e).lower():
                logger.error(f"Catalog error: {e}")
            raise
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def fetch_one(self, query: str, params: Optional[List[Any]] = None) -> Optional[tuple]:
        """Execute query and fetch one result"""
        result = self.execute(query, params)
        return result.fetchone() if result else None
    
    def fetch_all(self, query: str, params: Optional[List[Any]] = None) -> List[tuple]:
        """Execute query and fetch all results"""
        result = self.execute(query, params)
        return result.fetchall() if result else []
    
    def insert_returning(self, table: str, columns: List[str], values: List[Any], 
                        returning: str = "id") -> Optional[Any]:
        """Insert a row and return generated ID or specified column"""
        placeholders = ", ".join(["?" for _ in values])
        column_list = ", ".join(columns)
        
        query = f"""
            INSERT INTO {table} ({column_list})
            VALUES ({placeholders})
            RETURNING {returning}
        """
        
        result = self.fetch_one(query, values)
        return result[0] if result else None
    
    def batch_insert(self, table: str, columns: List[str], rows: List[List[Any]]) -> int:
        """Batch insert multiple rows efficiently"""
        if not rows:
            return 0
        
        placeholders = ", ".join(["?" for _ in columns])
        column_list = ", ".join(columns)
        
        query = f"""
            INSERT INTO {table} ({column_list})
            VALUES ({placeholders})
        """
        
        with self.transaction() as conn:
            inserted = 0
            for row in rows:
                conn.execute(query, row)
                inserted += 1
            return inserted
    
    def close(self):
        """Close the thread-local connection"""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()