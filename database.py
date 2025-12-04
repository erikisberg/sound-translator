"""
Database connection and initialization for Neon DB (serverless Postgres).
"""

import os
import streamlit as st
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager


def get_database_url() -> str | None:
    """Get database URL from Streamlit secrets or environment variables."""
    # Try Streamlit secrets first
    try:
        return st.secrets.get("database", {}).get("url")
    except Exception:
        pass

    # Fall back to environment variable
    return os.getenv("DATABASE_URL")


@st.cache_resource
def get_connection_pool():
    """
    Get a cached database connection.
    Uses Streamlit's cache_resource to maintain connection across reruns.
    """
    database_url = get_database_url()
    if not database_url:
        return None

    try:
        # Add connection timeout to prevent hanging
        conn = psycopg2.connect(
            database_url,
            cursor_factory=RealDictCursor,
            connect_timeout=10
        )
        conn.autocommit = True
        return conn
    except Exception as e:
        # Log error but don't block the app
        print(f"Database connection failed: {e}")
        return None


@contextmanager
def get_db_cursor():
    """
    Context manager for database operations.
    Automatically handles connection and cursor lifecycle.
    """
    conn = get_connection_pool()
    if conn is None:
        yield None
        return

    try:
        # Check if connection is still alive
        conn.poll()
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        # Connection lost, clear cache and reconnect
        get_connection_pool.clear()
        conn = get_connection_pool()
        if conn is None:
            yield None
            return

    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        cursor.close()


def init_db() -> bool:
    """
    Initialize the database schema.
    Creates the sessions table if it doesn't exist.
    Returns True if successful, False otherwise.
    """
    with get_db_cursor() as cursor:
        if cursor is None:
            return False

        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    name VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    status VARCHAR(50) DEFAULT 'draft',
                    source_filename VARCHAR(255),
                    source_duration_seconds FLOAT,
                    settings JSONB,
                    segments JSONB NOT NULL DEFAULT '[]'
                );
            """)

            # Create index for faster listing
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
                ON sessions(updated_at DESC);
            """)

            return True
        except Exception as e:
            st.error(f"Failed to initialize database: {e}")
            return False


def is_db_available() -> bool:
    """Check if database connection is available."""
    database_url = get_database_url()
    if not database_url:
        return False

    with get_db_cursor() as cursor:
        if cursor is None:
            return False
        try:
            cursor.execute("SELECT 1")
            return True
        except Exception:
            return False
