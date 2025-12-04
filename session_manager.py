"""
Session management for the Swedish Audio Translator.
Handles saving, loading, and managing translation sessions in Neon DB.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import streamlit as st
from database import get_db_cursor, init_db, is_db_available


@dataclass
class Session:
    """Represents a translation session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "draft"  # draft, in_progress, completed
    source_filename: Optional[str] = None
    source_duration_seconds: Optional[float] = None
    settings: dict = field(default_factory=dict)
    segments: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert session to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create session from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            name=data.get("name", ""),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            status=data.get("status", "draft"),
            source_filename=data.get("source_filename"),
            source_duration_seconds=data.get("source_duration_seconds"),
            settings=data.get("settings", {}),
            segments=data.get("segments", []),
        )

    @classmethod
    def from_db_row(cls, row: dict) -> "Session":
        """Create session from database row."""
        return cls(
            id=str(row["id"]),
            name=row["name"],
            created_at=row["created_at"].isoformat() if row["created_at"] else None,
            updated_at=row["updated_at"].isoformat() if row["updated_at"] else None,
            status=row["status"] or "draft",
            source_filename=row.get("source_filename"),
            source_duration_seconds=row.get("source_duration_seconds"),
            settings=row.get("settings") or {},
            segments=row.get("segments") or [],
        )


class SessionManager:
    """Manages session persistence in Neon DB."""

    def __init__(self):
        """Initialize session manager and ensure database is ready."""
        self._db_initialized = False

    def _ensure_db(self) -> bool:
        """Ensure database is initialized."""
        if not self._db_initialized:
            if is_db_available():
                self._db_initialized = init_db()
            else:
                return False
        return self._db_initialized

    def save_session(self, session: Session) -> bool:
        """
        Save or update a session in the database.
        Uses upsert (INSERT ... ON CONFLICT UPDATE).
        """
        if not self._ensure_db():
            return False

        session.updated_at = datetime.utcnow().isoformat()

        with get_db_cursor() as cursor:
            if cursor is None:
                return False

            try:
                cursor.execute(
                    """
                    INSERT INTO sessions (id, name, created_at, updated_at, status,
                                         source_filename, source_duration_seconds,
                                         settings, segments)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        updated_at = EXCLUDED.updated_at,
                        status = EXCLUDED.status,
                        source_filename = EXCLUDED.source_filename,
                        source_duration_seconds = EXCLUDED.source_duration_seconds,
                        settings = EXCLUDED.settings,
                        segments = EXCLUDED.segments
                    """,
                    (
                        session.id,
                        session.name,
                        session.created_at,
                        session.updated_at,
                        session.status,
                        session.source_filename,
                        session.source_duration_seconds,
                        json.dumps(session.settings),
                        json.dumps(session.segments),
                    ),
                )
                return True
            except Exception as e:
                st.error(f"Failed to save session: {e}")
                return False

    def load_session(self, session_id: str) -> Optional[Session]:
        """Load a session by ID."""
        if not self._ensure_db():
            return None

        with get_db_cursor() as cursor:
            if cursor is None:
                return None

            try:
                cursor.execute(
                    "SELECT * FROM sessions WHERE id = %s",
                    (session_id,),
                )
                row = cursor.fetchone()
                if row:
                    return Session.from_db_row(row)
                return None
            except Exception as e:
                st.error(f"Failed to load session: {e}")
                return None

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """
        List all sessions with metadata (without full segments).
        Returns list of dicts with: id, name, updated_at, status, segment_count
        """
        if not self._ensure_db():
            return []

        with get_db_cursor() as cursor:
            if cursor is None:
                return []

            try:
                cursor.execute(
                    """
                    SELECT id, name, created_at, updated_at, status,
                           source_filename, source_duration_seconds,
                           settings,
                           jsonb_array_length(segments) as segment_count
                    FROM sessions
                    ORDER BY updated_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "id": str(row["id"]),
                        "name": row["name"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                        "status": row["status"],
                        "source_filename": row["source_filename"],
                        "source_duration_seconds": row["source_duration_seconds"],
                        "segment_count": row["segment_count"] or 0,
                        "settings": row["settings"] or {},
                    }
                    for row in rows
                ]
            except Exception as e:
                st.error(f"Failed to list sessions: {e}")
                return []

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID."""
        if not self._ensure_db():
            return False

        with get_db_cursor() as cursor:
            if cursor is None:
                return False

            try:
                cursor.execute(
                    "DELETE FROM sessions WHERE id = %s",
                    (session_id,),
                )
                return cursor.rowcount > 0
            except Exception as e:
                st.error(f"Failed to delete session: {e}")
                return False

    def update_session_name(self, session_id: str, new_name: str) -> bool:
        """Update just the session name."""
        if not self._ensure_db():
            return False

        with get_db_cursor() as cursor:
            if cursor is None:
                return False

            try:
                cursor.execute(
                    """
                    UPDATE sessions
                    SET name = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (new_name, session_id),
                )
                return cursor.rowcount > 0
            except Exception as e:
                st.error(f"Failed to update session name: {e}")
                return False


def export_session_json(session: Session) -> str:
    """Export session as JSON string for download."""
    export_data = session.to_dict()
    export_data["_export_version"] = 1
    export_data["_exported_at"] = datetime.utcnow().isoformat()
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def import_session_json(json_data: str) -> Optional[Session]:
    """
    Import session from JSON string.
    Generates a new ID to avoid conflicts.
    """
    try:
        data = json.loads(json_data)
        # Generate new ID to avoid conflicts
        data["id"] = str(uuid.uuid4())
        data["created_at"] = datetime.utcnow().isoformat()
        data["updated_at"] = datetime.utcnow().isoformat()
        # Mark as imported
        if data.get("name"):
            data["name"] = f"{data['name']} (imported)"
        return Session.from_dict(data)
    except Exception as e:
        st.error(f"Failed to import session: {e}")
        return None


def generate_session_name(filename: Optional[str] = None) -> str:
    """Generate a default session name based on filename and date."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    if filename:
        # Remove extension and use as base
        base_name = filename.rsplit(".", 1)[0]
        return f"{base_name}_{date_str}"
    return f"Session_{date_str}"


# Singleton instance
@st.cache_resource
def get_session_manager() -> SessionManager:
    """Get cached session manager instance."""
    return SessionManager()
