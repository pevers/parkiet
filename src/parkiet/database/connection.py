import os
import psycopg
from contextlib import contextmanager
import logging

log = logging.getLogger(__name__)


class DatabaseConnection:
    """Database connection manager for PostgreSQL."""

    def __init__(self, connection_string: str | None = None):
        """
        Initialize database connection.

        Args:
            connection_string: PostgreSQL connection string. If None, will use environment variables.
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            # Use environment variables for connection
            self.connection_string = (
                f"postgresql://{os.getenv('DB_USER', 'parkiet')}:"
                f"{os.getenv('DB_PASSWORD', '')}@"
                f"{os.getenv('DB_HOST', 'localhost')}:"
                f"{os.getenv('DB_PORT', '5432')}/"
                f"{os.getenv('DB_NAME', 'parkiet')}"
            )

    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = None
        try:
            conn = psycopg.connect(
                self.connection_string, row_factory=psycopg.rows.dict_row
            )
            yield conn
        except Exception as e:
            log.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    @contextmanager
    def get_cursor(self):
        """Get a database cursor with automatic cleanup."""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                try:
                    yield cursor
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise


def get_db_connection() -> DatabaseConnection:
    """
    Factory function to get a database connection.
    """
    return DatabaseConnection()
