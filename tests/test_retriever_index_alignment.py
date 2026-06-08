from types import SimpleNamespace

import pytest

from src.rag_module_flash import SimpleRetriever


np = pytest.importorskip("numpy")


def _build_retriever() -> SimpleRetriever:
    retriever = SimpleRetriever.__new__(SimpleRetriever)
    retriever.config = SimpleNamespace(neighbor_window=0)
    retriever.documents = []
    retriever.document_embeddings = []
    retriever.chunk_lookup = {}
    retriever.embeddings_model = None
    return retriever


def test_cached_duplicate_filter_keeps_embeddings_aligned():
    retriever = _build_retriever()
    retriever.documents = [{"content": "already indexed", "metadata": {"chunk_id": 0}}]
    retriever.document_embeddings = np.array([[1.0, 0.0]])

    cached_docs = [
        {"content": "already indexed", "metadata": {"chunk_id": 0}},
        {"content": "new cached chunk", "metadata": {"chunk_id": 1}},
    ]
    cached_embeddings = np.array([[9.0, 9.0], [0.0, 1.0]])

    new_docs, new_embeddings = retriever._select_new_docs_with_embeddings(
        cached_docs,
        cached_embeddings,
        {"already indexed"},
    )
    retriever._append_documents_with_embeddings(new_docs, new_embeddings)

    assert [doc["content"] for doc in retriever.documents] == [
        "already indexed",
        "new cached chunk",
    ]
    assert retriever.document_embeddings.shape == (2, 2)
    assert retriever.document_embeddings.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_append_without_embeddings_disables_stale_semantic_index():
    retriever = _build_retriever()
    retriever.documents = [{"content": "first", "metadata": {}}]
    retriever.document_embeddings = np.array([[1.0, 0.0]])

    retriever._append_documents_with_embeddings([{"content": "second", "metadata": {}}])

    assert len(retriever.documents) == 2
    assert retriever.document_embeddings == []
