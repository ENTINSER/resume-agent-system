"""RAG 检索策略测试"""

import pytest

from src.vector_store.models import VectorDocument
from src.vector_store.retrieval import RAGRetriever, validate_doc_type
from src.vector_store.vector_store import VectorStore
from src.vector_store.embedding_service import EmbeddingService


@pytest.fixture
def retrieval(mock_embedding_service):
    store = VectorStore(
        url=":memory:",
        collection_name="test_retrieval",
        vector_size=EmbeddingService.VECTOR_SIZE,
        embedding_service=mock_embedding_service,
    )
    store.upsert([
        VectorDocument(
            id="p1",
            doc_type="project_summary",
            session_id="s1",
            task_id="t1",
            content="项目: Calculator\n需求: Python 计算器",
            metadata={"project_name": "Calculator"},
        ),
        VectorDocument(
            id="c1",
            doc_type="code",
            session_id="s1",
            task_id="t1",
            content="def add(a, b): return a + b",
            metadata={"file_path": "main.py"},
        ),
        VectorDocument(
            id="c2",
            doc_type="external_code",
            session_id="",
            task_id=None,
            content="def subtract(a, b): return a - b",
            metadata={"repo_name": "example"},
        ),
        VectorDocument(
            id="e1",
            doc_type="evaluation",
            session_id="s1",
            task_id="t1",
            content="Calculator 评估总分 90",
            metadata={"overall_score": 90},
        ),
        VectorDocument(
            id="f1",
            doc_type="test_failure",
            session_id="s1",
            task_id="t1",
            content="ZeroDivisionError in divide function",
            metadata={"file": "test_divide.py"},
        ),
    ])
    return RAGRetriever(vector_store=store)


class TestRAGRetriever:
    def test_find_similar_projects_prioritizes_summary(self, retrieval):
        results = retrieval.find_similar_projects("Python 计算器", top_k=2)
        assert len(results) >= 1
        assert results[0]["doc_type"] == "project_summary"

    def test_find_similar_code_includes_external(self, retrieval):
        results = retrieval.find_similar_code("subtraction", top_k=2)
        doc_types = {r["doc_type"] for r in results}
        assert "code" in doc_types or "external_code" in doc_types

    def test_find_similar_evaluations(self, retrieval):
        results = retrieval.find_similar_evaluations("Calculator score", top_k=2)
        assert len(results) >= 1
        assert any(r["doc_type"] == "evaluation" for r in results)

    def test_find_failure_patterns(self, retrieval):
        results = retrieval.find_failure_patterns("ZeroDivisionError", top_k=2)
        assert len(results) >= 1
        assert results[0]["doc_type"] == "test_failure"

    def test_multi_step_retrieve(self, retrieval):
        results = retrieval.multi_step_retrieve("Python 计算器", top_k_per_step=2)
        assert "projects" in results
        assert "code" in results
        assert "evaluations" in results
        assert len(results["projects"]) >= 1

    def test_retrieve_for_generation(self, retrieval):
        ctx = retrieval.retrieve_for_generation("Python 计算器", top_k=2)
        assert "similar_projects" in ctx
        assert "similar_code" in ctx
        assert "failure_patterns" in ctx


class TestValidateDocType:
    def test_known_type(self):
        assert validate_doc_type("code") == "code"

    def test_unknown_type(self):
        assert validate_doc_type("unknown_type") is None

    def test_none(self):
        assert validate_doc_type(None) is None
