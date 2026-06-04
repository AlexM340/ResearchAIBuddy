"""
Postgres client wrapper used by the Second Brain storage layer.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

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
            logger.warning("Conexiunea Postgres a esuat: %s", exc)
            return False

    def initialize_schema(self, schema_path: Optional[str] = None) -> bool:
        """Initialize schema from SQL file, then apply additive migrations."""
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
            return self.run_migrations()
        except Exception as exc:
            logger.error("Initializarea schemei Postgres a esuat: %s", exc)
            return False

    def run_migrations(self, migrations_dir: Optional[str] = None) -> bool:
        """Apply SQL migrations once, tracked in schema_migrations."""
        if not self.enabled:
            return False

        try:
            migration_path = (
                Path(migrations_dir)
                if migrations_dir
                else Path(__file__).parent / "migrations"
            )
            if not migration_path.exists():
                return True

            migration_files = sorted(path for path in migration_path.glob("*.sql") if path.is_file())
            with self.connection() as conn:
                if conn is None:
                    return False
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS schema_migrations (
                            version TEXT PRIMARY KEY,
                            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        """
                    )
                    cur.execute("SELECT version FROM schema_migrations;")
                    applied = {str(row[0]) for row in (cur.fetchall() or [])}

                    for file_path in migration_files:
                        version = file_path.name
                        if version in applied:
                            continue
                        cur.execute(file_path.read_text(encoding="utf-8"))
                        cur.execute(
                            """
                            INSERT INTO schema_migrations(version, applied_at)
                            VALUES (%s, NOW())
                            ON CONFLICT(version) DO NOTHING;
                            """,
                            (version,),
                        )
                        logger.info("Applied Postgres migration %s", version)
                conn.commit()
            return True
        except Exception as exc:
            logger.error("Aplicarea migratiilor Postgres a esuat: %s", exc)
            return False

    def get_migration_status(self, migrations_dir: Optional[str] = None) -> Dict[str, Any]:
        """Return applied/pending migration names for UI diagnostics."""
        status: Dict[str, Any] = {"enabled": self.enabled, "applied": [], "pending": [], "error": None}
        if not self.enabled:
            return status

        try:
            migration_path = (
                Path(migrations_dir)
                if migrations_dir
                else Path(__file__).parent / "migrations"
            )
            known = sorted(path.name for path in migration_path.glob("*.sql")) if migration_path.exists() else []
            with self.connection() as conn:
                if conn is None:
                    status["error"] = "connection_unavailable"
                    return status
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS schema_migrations (
                            version TEXT PRIMARY KEY,
                            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        );
                        """
                    )
                    cur.execute("SELECT version FROM schema_migrations ORDER BY version;")
                    applied = [str(row[0]) for row in (cur.fetchall() or [])]
                conn.commit()
            status["applied"] = applied
            status["pending"] = [version for version in known if version not in set(applied)]
            return status
        except Exception as exc:
            status["error"] = str(exc)
            return status
