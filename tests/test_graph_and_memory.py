from __future__ import annotations

from types import MethodType, SimpleNamespace

from src.graph.graph_ingestion import GraphIngestionService
from src.rag_module_flash import APCISystem


class DummyNeo4jClient:
    def __init__(self) -> None:
        self.enabled = True
        self.writes = []
        self.reads = []

    def run_write(self, cypher, parameters=None):
        self.writes.append((cypher, parameters or {}))
        return True

    def run_read(self, cypher, parameters=None):
        self.reads.append((cypher, parameters or {}))
        if "DETACH DELETE" in cypher:
            return [{"count": 0}]
        return []


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


def test_citation_validator_strips_unbacked_citations():
    apci = _build_test_apci()
    validation = apci._validate_citations(
        "Raspuns valid [D1] dar citatia asta nu exista [N9].",
        {"D1": {"kind": "document"}},
    )

    assert "[D1]" in validation["response"]
    assert "[N9]" not in validation["response"]
    assert validation["invalid"] == ["N9"]
    assert validation["used"] == ["D1"]


def test_medium_confidence_task_becomes_persistent_proposal():
    apci = _build_test_apci()

    class DummyRepository:
        enabled = True

        def __init__(self):
            self.created = []

        def create_memory_proposal(self, proposal_type, payload, confidence, topic_collection):
            self.created.append((proposal_type, payload, confidence, topic_collection))
            return 42

    repo = DummyRepository()
    apci.repository = repo
    persisted = []
    apci._persist_tasks = MethodType(lambda self, tasks: persisted.extend(tasks), apci)

    outcome = apci._process_memory_candidates(
        "task",
        [
            {
                "title": "citeste sursele noi",
                "details": "detalii",
                "topic_collection": "licenta",
                "priority": "normal",
                "confidence": 0.78,
            }
        ],
    )

    assert persisted == []
    assert outcome["auto_saved"] == []
    assert outcome["proposed"][0]["proposal_id"] == 42
    assert repo.created[0][0] == "task"


def test_topic_analysis_prompt_builds_typed_citation_map():
    apci = _build_test_apci()
    prompt, citation_map = apci._build_topic_analysis_prompt(
        "taxonomy",
        "licenta",
        {
            "doc_chunks": [{"content": "GraphRAG combina vectori si graf.", "metadata": {"filename": "paper.md"}}],
            "notes": [{"id": 7, "title": "Idee", "content": "Prefer explicatii scurte."}],
            "decisions": [{"id": 3, "title": "Ruta hybrid", "rationale": "Folosim hybrid pentru relatii."}],
            "preferences": [{"id": 2, "preference_key": "language", "preference_value": "romana"}],
            "memory_artifacts": [{"id": 9, "artifact_type": "task", "title": "Task", "content": "Verifica logs."}],
            "graph_items": [{"source": "GraphRAG", "target": "Retrieval", "type": "RELATED_TO"}],
        },
    )

    assert "[D1]" in prompt
    assert "[N1]" in prompt
    assert "[Dec1]" in prompt
    assert "[P1]" in prompt
    assert "[M1]" in prompt
    assert "[G1]" in prompt
    assert set(citation_map) == {"D1", "N1", "Dec1", "P1", "M1", "G1"}
