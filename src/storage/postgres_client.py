"""
Postgres client wrapper used by the Second Brain storage layer.
"""

from __future__ import annotations

import logging
import threading
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

try:
    from psycopg_pool import ConnectionPool

    PSYCOPG_POOL_AVAILABLE = True
except Exception:
    ConnectionPool = None
    PSYCOPG_POOL_AVAILABLE = False


class PostgresClient:
    """Thin wrapper over psycopg with optional runtime enablement."""

    def __init__(
        self,
        dsn: str = "",
        embedding_dim: int = 384,
        enabled: bool = True,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
    ):
        self.dsn = (dsn or "").strip()
        self.embedding_dim = int(embedding_dim or 384)
        self.enabled = bool(enabled and self.dsn and PSYCOPG_AVAILABLE)
        self.pool_min_size = int(pool_min_size)
        self.pool_max_size = int(pool_max_size)

        # Lazily-created connection pool, guarded for thread-safe init.
        self._pool: Optional["ConnectionPool"] = None
        self._pool_lock = threading.Lock()

        if enabled and not PSYCOPG_AVAILABLE:
            logger.warning("psycopg nu este disponibil; storage Postgres dezactivat.")
        if enabled and not self.dsn:
            logger.info("storage.postgres_dsn nu este configurat; storage Postgres dezactivat.")

    def _get_pool(self) -> Optional["ConnectionPool"]:
        """Return a lazily-initialized connection pool, or None if unavailable.

        Pooling avoids a fresh TCP+TLS handshake on every repository call.
        Falls back transparently to one-shot connections when psycopg_pool
        is not installed, so behaviour is unchanged in that case.
        """
        if not self.enabled or not PSYCOPG_POOL_AVAILABLE:
            return None
        if self._pool is not None:
            return self._pool
        with self._pool_lock:
            if self._pool is None:
                try:
                    self._pool = ConnectionPool(
                        conninfo=self.dsn,
                        min_size=self.pool_min_size,
                        max_size=self.pool_max_size,
                        # prepare_threshold=None dezactiveaza prepared statements:
                        # obligatoriu cu poolere in mod tranzactie (Supabase/PgBouncer pe
                        # 6543), altfel apar erori "prepared statement _pgN already exists"
                        # cand conexiunile sunt reutilizate prin pooler.
                        kwargs={"autocommit": False, "prepare_threshold": None},
                        open=True,
                    )
                except Exception as exc:
                    logger.warning("Connection pool init failed, using one-shot connections: %s", exc)
                    self._pool = None
        return self._pool

    @contextmanager
    def connection(self) -> Generator[Optional["psycopg.Connection"], None, None]:
        """Yield a pooled connection if Postgres is enabled.

        The connection is borrowed from the pool for the duration of the
        ``with`` block and returned afterwards (not physically closed). When
        no pool is available it degrades to a single-use connection, matching
        the previous behaviour exactly.
        """
        if not self.enabled:
            yield None
            return

        pool = self._get_pool()
        if pool is not None:
            with pool.connection() as conn:
                yield conn
            return

        # Fallback: one-shot connection (no pooling available).
        conn = None
        try:
            conn = psycopg.connect(self.dsn, autocommit=False, prepare_threshold=None)
            yield conn
        finally:
            if conn is not None:
                conn.close()

    def close(self) -> None:
        """Dispose the connection pool, if any. Safe to call multiple times."""
        with self._pool_lock:
            if self._pool is not None:
                try:
                    self._pool.close()
                except Exception as exc:
                    logger.warning("Connection pool close failed: %s", exc)
                finally:
                    self._pool = None

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
