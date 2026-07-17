"""RAGRetriever 单元测试"""

import pytest

from src.vector_store.rag_retriever import RAGRetriever
from src.vector_store.vector_store import VectorStore
from src.vector_store.embedding_service import EmbeddingService
from src.vector_store.models import VectorDocument


@pytest.fixture
def rag_retriever(mock_embedding_service):
    store = VectorStore(
        url=":memory:",
        collection_name="test_rag",
        vector_size=EmbeddingService.VECTOR_SIZE,
        embedding_service=mock_embedding_service,
    )
    # 预置数据
    store.upsert([
        VectorDocument(
            id="p1",
            doc_type="project_summary",
            session_id="s1",
            task_id="t1",
            content="项目: Calculator\n技术栈: pytest\n需求: 开发一个Python计算器",
            metadata={"project_name": "Calculator", "overall_score": 90},
        ),
        VectorDocument(
            id="c1",
            doc_type="code",
            session_id="s1",
            task_id="t1",
            content="文件: main.py\n\ndef add(a, b): return a + b",
            metadata={"file_path": "main.py"},
        ),
        VectorDocument(
            id="e1",
            doc_type="evaluation",
            session_id="s1",
            task_id="t1",
            content="项目: Calculator\n评分: 90\n通过: True",
            metadata={"overall_score": 90, "passed": True},
        ),
    ])
    return RAGRetriever(vector_store=store)


class TestRAGRetriever:
    def test_find_similar_projects(self, rag_retriever):
        results = rag_retriever.find_similar_projects("Python 计算器", top_k=3)
        assert len(results) >= 1
        assert results[0]["doc_type"] == "project_summary"

    def test_find_similar_code(self, rag_retriever):
        results = rag_retriever.find_similar_code("加法函数", top_k=3)
        assert len(results) >= 1
        assert results[0]["doc_type"] == "code"

    def test_find_similar_evaluations(self, rag_retriever):
        results = rag_retriever.find_similar_evaluations("计算器评估", top_k=3)
        assert len(results) >= 1
        assert results[0]["doc_type"] == "evaluation"

    def test_format_context(self, rag_retriever):
        results = [{"score": 0.95, "content": "测试", "metadata": {"project_name": "Test"}}]
        ctx = rag_retriever.format_context(results)
        assert "相似历史案例" in ctx
        assert "Test" in ctx
