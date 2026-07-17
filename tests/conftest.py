"""测试公共 fixture"""

import pytest
from pydantic_settings import SettingsConfigDict


class _TestSettings:
    """占位，实际在下面 fixture 中动态替换"""


class MockEmbeddingService:
    """Mock 嵌入服务，避免下载真实模型"""

    def __init__(self, vector_size: int = 384):
        self.vector_size = vector_size

    def encode(self, texts):
        import hashlib
        results = []
        for text in texts:
            # 根据文本内容生成确定性向量
            h = hashlib.md5(text.encode("utf-8")).digest()
            vec = []
            for i in range(self.vector_size):
                vec.append(((h[i % len(h)] + i) % 200 - 100) / 100.0)
            results.append(vec)
        return results

    def encode_single(self, text):
        return self.encode([text])[0]


@pytest.fixture
def mock_embedding_service():
    return MockEmbeddingService()


@pytest.fixture(autouse=True)
def deterministic_settings(monkeypatch):
    """测试期间忽略 .env 文件，确保配置默认值稳定"""
    from src.core.config import Settings

    class TestSettings(Settings):
        model_config = SettingsConfigDict(
            env_file=None,
            env_file_encoding="utf-8",
            populate_by_name=True,
            extra="ignore",
        )

    monkeypatch.setattr("src.core.config.Settings", TestSettings)
