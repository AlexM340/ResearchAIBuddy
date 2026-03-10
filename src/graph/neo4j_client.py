"""
Neo4j client wrapper for graph ingestion and querying.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except Exception:
    GraphDatabase = None
    NEO4J_AVAILABLE = False


class Neo4jClient:
    """Simple neo4j wrapper with graceful disable mode."""

    def __init__(self, uri: str = "", user: str = "", password: str = "", enabled: bool = True):
        self.uri = (uri or "").strip()
        self.user = (user or "").strip()
        self.password = (password or "").strip()
        self.enabled = bool(enabled and self.uri and self.user and self.password and NEO4J_AVAILABLE)
        self._driver = None

        if enabled and not NEO4J_AVAILABLE:
            logger.warning("neo4j driver nu este disponibil; graph layer dezactivat.")

        if self.enabled:
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            except Exception as exc:
                logger.warning("Neo4j init failed, graph layer disabled: %s", exc)
                self.enabled = False
                self._driver = None

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()

    def ensure_constraints(self) -> bool:
        if not self.enabled or self._driver is None:
            return False
        try:
            statements = [
                "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT decision_id_unique IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
                "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE",
                "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            ]
            with self._driver.session() as session:
                for stmt in statements:
                    session.run(stmt)
            return True
        except Exception as exc:
            logger.warning("Neo4j constraint init failed: %s", exc)
            return False

    def run_write(self, cypher: str, parameters: Dict[str, Any] | None = None) -> bool:
        if not self.enabled or self._driver is None:
            return False
        try:
            with self._driver.session() as session:
                session.run(cypher, parameters or {})
            return True
        except Exception as exc:
            logger.error("Neo4j write failed: %s", exc)
            return False

    def run_read(self, cypher: str, parameters: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        if not self.enabled or self._driver is None:
            return []
        try:
            with self._driver.session() as session:
                result = session.run(cypher, parameters or {})
                return [record.data() for record in result]
        except Exception as exc:
            logger.error("Neo4j read failed: %s", exc)
            return []

