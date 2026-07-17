"""EmbeddingService 单元测试

真实模型测试需要下载 sentence-transformers 模型，默认使用 Mock 避免网络依赖。
"""

import pytest

from src.vector_store.embedding_service import EmbeddingService


@pytest.fixture
def embedding_service():
    # 使用 tiny 模型或本地路径，避免测试时下载大模型
    # 如果环境变量指定了模型则使用，否则用 all-MiniLM-L6-v2（首次会下载）
    import os
    model = os.getenv("TEST_EMBEDDING_MODEL", EmbeddingService.DEFAULT_MODEL)
    return EmbeddingService(model_name=model)


@pytest.mark.slow
class TestEmbeddingService:
    def test_encode_single(self, embedding_service):
        vector = embedding_service.encode_single("hello world")
        assert isinstance(vector, list)
        assert len(vector) == EmbeddingService.VECTOR_SIZE
        assert all(isinstance(v, float) for v in vector)

    def test_encode_batch(self, embedding_service):
        vectors = embedding_service.encode(["hello", "world", "hello"])
        assert len(vectors) == 3
        # 缓存命中：相同文本应返回相同向量
        assert vectors[0] == vectors[2]

    def test_empty_input(self, embedding_service):
        assert embedding_service.encode([]) == []

    def test_encode_all_cached(self, embedding_service):
        # 先编码一次，使文本进入缓存
        embedding_service.encode(["cached text"])
        # 再次编码相同文本：全部命中缓存，不应 KeyError
        vectors = embedding_service.encode(["cached text", "cached text"])
        assert len(vectors) == 2
        assert vectors[0] == vectors[1]
