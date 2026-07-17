"""VectorStore 单元测试

使用 Qdrant 内存模式，无需外部服务。
"""

import pytest

from src.vector_store.models import VectorDocument
from src.vector_store.vector_store import VectorStore
from src.vector_store.embedding_service import EmbeddingService


@pytest.fixture
def vector_store(mock_embedding_service):
    """创建内存 VectorStore"""
    return VectorStore(
        url=":memory:",
        collection_name="test_memory",
        vector_size=EmbeddingService.VECTOR_SIZE,
        embedding_service=mock_embedding_service,
    )


class TestVectorStore:
    def test_create_collection(self, vector_store):
        assert vector_store.count() == 0

    def test_upsert_and_search(self, vector_store):
        docs = [
            VectorDocument(
                id="doc-1",
                doc_type="requirement",
                session_id="s1",
                task_id="t1",
                content="开发一个 Python 计算器",
                metadata={"project_name": "Calculator"},
            ),
            VectorDocument(
                id="doc-2",
                doc_type="code",
                session_id="s1",
                task_id="t1",
                content="def add(a, b): return a + b",
                metadata={"file_path": "main.py"},
            ),
        ]
        vector_store.upsert(docs)
        assert vector_store.count() == 2

        results = vector_store.search("计算器", top_k=5)
        assert len(results) >= 1
        assert results[0]["doc_type"] == "requirement"

    def test_search_with_filter(self, vector_store):
        docs = [
            VectorDocument(
                id="doc-1",
                doc_type="requirement",
                session_id="s1",
                task_id="t1",
                content="计算器需求",
            ),
            VectorDocument(
                id="doc-2",
                doc_type="code",
                session_id="s2",
                task_id="t2",
                content="计算器代码",
            ),
        ]
        vector_store.upsert(docs)

        results = vector_store.search("计算器", doc_type="code", top_k=5)
        assert len(results) == 1
        assert results[0]["doc_type"] == "code"

    def test_delete_by_session(self, vector_store):
        docs = [
            VectorDocument(
                id="doc-1",
                doc_type="requirement",
                session_id="s1",
                task_id="t1",
                content="需求1",
            ),
            VectorDocument(
                id="doc-2",
                doc_type="requirement",
                session_id="s2",
                task_id="t2",
                content="需求2",
            ),
        ]
        vector_store.upsert(docs)
        assert vector_store.count() == 2

        vector_store.delete_by_session("s1")
        assert vector_store.count() == 1
