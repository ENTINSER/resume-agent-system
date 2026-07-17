"""文本嵌入服务

使用 sentence-transformers 本地模型将文本编码为向量。
支持批量编码和简单缓存。
"""

import hashlib
import os
from collections import OrderedDict
from typing import List, Optional

from src.core.logger import logger
from src.core.config import settings


# 国内网络环境下 HuggingFace 官方源经常超时，默认使用镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 避免默认 10s 超时 + 指数退避导致长时间无响应
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")


class EmbeddingModelUnavailableError(RuntimeError):
    """嵌入模型无法加载"""
    pass


class EmbeddingService:
    """文本嵌入服务"""

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    VECTOR_SIZE = 384
    DEFAULT_CACHE_SIZE = 10000

    def __init__(self, model_name: Optional[str] = None, cache_size: Optional[int] = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._cache_size = cache_size if cache_size is not None else self.DEFAULT_CACHE_SIZE
        self._cache: OrderedDict[str, list] = OrderedDict()

    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            logger.info(f"[Embedding] 加载模型: {self.model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except Exception as e:
                logger.error(f"[Embedding] 模型加载失败: {e}")
                raise EmbeddingModelUnavailableError(
                    f"无法加载嵌入模型 {self.model_name}: {e}"
                )
        return self._model

    def encode(self, texts: List[str]) -> List[List[float]]:
        """批量编码文本为向量"""
        if not texts:
            return []

        # 去重 + 缓存命中
        unique_texts = []
        text_to_idx = {}
        for t in texts:
            key = self._cache_key(t)
            if key in self._cache:
                continue
            if key not in text_to_idx:
                text_to_idx[key] = len(unique_texts)
                unique_texts.append(t)

        # 已缓存的结果直接复用，未缓存的批量编码
        key_to_embedding: dict[str, list] = {
            self._cache_key(t): self._cache[self._cache_key(t)]
            for t in texts
            if self._cache_key(t) in self._cache
        }
        if unique_texts:
            logger.debug(f"[Embedding] 编码 {len(unique_texts)} 条文本")
            embeddings = self.model.encode(unique_texts, convert_to_numpy=True)
            for key, idx in text_to_idx.items():
                vector = embeddings[idx].tolist()
                key_to_embedding[key] = vector
                self._cache[key] = vector
                self._cache.move_to_end(key)
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

        return [key_to_embedding[self._cache_key(t)] for t in texts]

    def encode_single(self, text: str) -> List[float]:
        """编码单条文本"""
        return self.encode([text])[0]

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
