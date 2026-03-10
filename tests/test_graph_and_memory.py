from __future__ import annotations

from types import MethodType, SimpleNamespace

from src.graph.graph_ingestion import GraphIngestionService
from src.rag_module_flash import APCISystem


class DummyNeo4jClient:
    def __init__(self) -> None:
        self.enabled = True
        self.writes = []

    def run_write(self, cypher, parameters=None):
        self.writes.append((cypher, parameters or {}))
        return True


def test_graph_ingestion_structured_mode_applies_confidence_gating():
    client = DummyNeo4jClient()

    def structured_extractor(_text: str):
        return {
            "concepts": [
                {"name": "GraphRAG", "confidence": 0.9},
                {"name": "Noise", "confidence": 0.2},
            ],
            "relations": [
                {
                    "source": "GraphRAG",
                    "target": "Retrieval",
                    "type": "DEPENDS_ON",
                    "confidence": 0.91,
                },
                {
                    "source": "GraphRAG",
                    "target": "Hallucination",
                    "type": "RELATED_TO",
                    "confidence": 0.3,
                },
            ],
        }

    service = GraphIngestionService(
        neo4j_client=client,
        min_confidence=0.70,
        extraction_mode="llm_structured",
        structured_extractor=structured_extractor,
    )
    result = service.ingest_chunks(
        [
            {
                "content": "GraphRAG depends on retrieval.",
                "metadata": {
                    "source_path": "d:/tmp/doc.txt",
                    "doc_id": "doc_1",
                    "filename": "doc.txt",
                    "collection": "general",
                    "chunk_id": 0,
                },
            }
        ]
    )

    assert result["chunks"] == 1
    assert result["concepts"] == 1
    assert result["relations"] == 1
    assert any("DEPENDS_ON" in cypher for cypher, _ in client.writes)


def _build_test_apci() -> APCISystem:
    apci = APCISystem.__new__(APCISystem)
    apci.config = SimpleNamespace(memory_consolidation_enabled=True, memory_decay_days=365)
    apci.local_decisions = []
    apci.local_preferences = []
    apci.local_episodes = []
    apci.local_tasks = []
    apci.last_memory_consolidation_at = None
    apci.repository = None
    apci.graph_ingestion = None
    return apci


def test_task_extraction_from_chat_marker():
    apci = _build_test_apci()
    tasks = apci._extract_task_candidates(
        question="Creeaza un task: finalizeaza capitolul 3 din lucrare",
        response="Am notat urmatorul pas.",
        active_collection="licenta",
    )

    assert len(tasks) == 1
    assert tasks[0]["memory_type"] == "task"
    assert "capitolul 3" in tasks[0]["title"]
    assert tasks[0]["topic_collection"] == "licenta"


def test_memory_consolidation_promotes_episode_to_semantic():
    apci = _build_test_apci()
    persisted = []

    def fake_persist_decision(self, decision):
        persisted.append(decision)
        self.local_decisions.append(decision)

    apci._persist_decision = MethodType(fake_persist_decision, apci)
    apci.local_episodes = [
        {
            "id": "episode_1",
            "title": "Arhitectura retrieval",
            "rationale": "Concluzie: folosim route hibrid pentru intrebari relationale.",
            "topic_collection": "licenta",
            "confidence": 0.7,
            "created_at": "2026-03-09T10:00:00+00:00",
            "updated_at": "2026-03-09T10:00:00+00:00",
        }
    ]

    apci._run_memory_consolidation(active_collection="licenta")

    assert len(persisted) == 1
    assert persisted[0]["memory_type"] == "semantic"
    assert persisted[0]["source"] == "memory_consolidation"
