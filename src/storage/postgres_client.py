"""
Postgres client wrapper used by the Second Brain storage layer.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg
    PSYCOPG_AVAILABLE = True
except Exception:
    psycopg = None
    PSYCOPG_AVAILABLE = False


class PostgresClient:
    """Thin wrapper over psycopg with optional runtime enablement."""

    def __init__(
        self,
        dsn: str = "",
        embedding_dim: int = 384,
        enabled: bool = True,
    ):
        self.dsn = (dsn or "").strip()
        self.embedding_dim = int(embedding_dim or 384)
        self.enabled = bool(enabled and self.dsn and PSYCOPG_AVAILABLE)

        if enabled and not PSYCOPG_AVAILABLE:
            logger.warning("psycopg nu este disponibil; storage Postgres dezactivat.")

        if enabled and not self.dsn:
            logger.info("storage.postgres_dsn nu este configurat; storage Postgres dezactivat.")

    @contextmanager
    def connection(self) -> Generator[Optional["psycopg.Connection"], None, None]:
        """Open a connection if Postgres is enabled."""
        if not self.enabled:
            yield None
            return

        conn = None
        try:
            conn = psycopg.connect(self.dsn, autocommit=False)
            yield conn
        finally:
            if conn is not None:
                conn.close()

    def test_connection(self) -> bool:
        """Validate DB connection and pgvector availability."""
        if not self.enabled:
            return False

        try:
            with self.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
                    cur.execute("SELECT extname FROM pg_extension WHERE extname='vector';")
                    _ = cur.fetchone()
                conn.commit()
            return True
        except Exception as exc:
            logger.warning("Conexiunea Postgres a eșuat: %s", exc)
            return False

    def initialize_schema(self, schema_path: Optional[str] = None) -> bool:
        """Initialize schema from SQL file."""
        if not self.enabled:
            return False

        try:
            schema_file = (
                Path(schema_path)
                if schema_path
                else Path(__file__).parent / "schema.sql"
            )
            sql_text = schema_file.read_text(encoding="utf-8")
            with self.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(sql_text)
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Inițializarea schemei Postgres a eșuat: %s", exc)
            return False

