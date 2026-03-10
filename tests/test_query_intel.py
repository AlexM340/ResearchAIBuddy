from src.query.hybrid_retriever import HybridEvidenceFusion
from src.query.intent_router import RuleBasedIntentRouter


def test_router_prefers_hybrid_for_relational_questions():
    router = RuleBasedIntentRouter(vector_confidence_threshold=0.35)
    intent = router.classify("De ce depinde performanta retrieval-ului?")
    plan = router.build_plan(
        intent=intent,
        retrieval_mode="topic_general",
        vector_top_k=8,
        graph_top_k_paths=6,
        hybrid_rerank_top_k=8,
        vector_confidence=0.9,
    )
    assert intent.intent_type == "relational"
    assert plan.route == "hybrid"


def test_router_escalates_when_vector_confidence_is_low():
    router = RuleBasedIntentRouter(vector_confidence_threshold=0.35)
    intent = router.classify("Care este definitia chunking?")
    plan = router.build_plan(
        intent=intent,
        retrieval_mode="topic_general",
        vector_top_k=8,
        graph_top_k_paths=6,
        hybrid_rerank_top_k=8,
        vector_confidence=0.2,
    )
    assert plan.route == "hybrid"


def test_hybrid_fusion_returns_ranked_evidence_and_provenance():
    fusion = HybridEvidenceFusion()
    vector_docs = [
        {
            "content": "Document evidence A",
            "metadata": {"filename": "a.txt"},
            "retrieval_score": 0.9,
            "semantic_score": 0.9,
            "keyword_score": 0.3,
        },
        {
            "content": "Document evidence B",
            "metadata": {"filename": "b.txt"},
            "retrieval_score": 0.5,
            "semantic_score": 0.5,
            "keyword_score": 0.1,
        },
    ]
    graph_paths = [
        {
            "content": "Graph evidence",
            "concept": "retrieval",
            "filename": "graph.md",
            "score": 0.8,
            "path": ["retrieval", "DEPENDS_ON", "chunking"],
        }
    ]

    evidence, provenance = fusion.fuse(vector_docs=vector_docs, graph_paths=graph_paths, rerank_top_k=3)
    assert len(evidence) == 3
    assert len(provenance) == 3
    assert evidence[0].score >= evidence[1].score
    assert provenance[0]["citation"] in {"D1", "D2", "G1"}
