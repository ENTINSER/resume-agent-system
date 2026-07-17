"""Qdrant 向量存储封装

提供 collection 管理、文档写入、向量检索能力。
"""

import os
import threading
import uuid
from typing import List, Optional, Dict, Any


def _normalize_point_id(point_id: str) -> str:
    """确保点 ID 是 Qdrant 可接受的 UUID 字符串"""
    try:
        uuid.UUID(point_id)
        return point_id
    except ValueError:
        # 用 ID 的 MD5 生成确定性 UUID
        import hashlib
        hex_str = hashlib.md5(point_id.encode("utf-8")).hexdigest()
        # 调整成标准 UUID 格式：8-4-4-4-12
        return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.core.logger import logger
from src.core.config import settings
from src.vector_store.models import VectorDocument
from src.vector_store.embedding_service import EmbeddingService


class VectorStore:
    """Qdrant 向量存储"""

    DEFAULT_COLLECTION = "agent_memory"
    DEFAULT_VECTOR_SIZE = 384

    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: int = DEFAULT_VECTOR_SIZE,
        embedding_service: Optional[EmbeddingService] = None,
        silent_fallback: bool = False,
        local_path: Optional[str] = None,
    ):
        self.url = url or settings.qdrant_url
        self.collection_name = collection_name or settings.qdrant_collection
        self.vector_size = vector_size
        self.embedding_service = embedding_service or EmbeddingService()
        self.local_path = local_path or getattr(settings, "qdrant_local_path", "data/qdrant_storage")

        self.client = self._create_client(silent_fallback)
        self._ensure_collection()

    def _create_client(self, silent_fallback: bool) -> QdrantClient:
        """创建 QdrantClient，支持远程、本地持久化、内存三种模式"""
        client_kwargs = {"check_compatibility": False}

        if self.url == ":memory:":
            logger.info("[VectorStore] 使用内存模式")
            return QdrantClient(location=":memory:", **client_kwargs)

        # 显式本地路径（以 / 开头或不含协议）
        if self.url.startswith("/") or (
            "://" not in self.url and "." not in self.url.split(os.sep)[-1]
        ):
            logger.info(f"[VectorStore] 使用本地持久化路径: {self.url}")
            return QdrantClient(path=self.url, **client_kwargs)

        # 默认按 HTTP(S) 远程连接尝试
        try:
            client = QdrantClient(url=self.url, **client_kwargs)
            # 做一次轻量健康检查，避免延迟到首次查询才发现连不上
            client.get_collections()
            logger.info(f"[VectorStore] 已连接远程 Qdrant: {self.url}")
            return client
        except Exception as e:
            if not silent_fallback:
                logger.warning(
                    f"[VectorStore] 无法连接远程 Qdrant ({self.url}): {e}"
                )

        # 回退到本地持久化
        try:
            if not silent_fallback:
                logger.warning(f"[VectorStore] 回退到本地持久化: {self.local_path}")
            return QdrantClient(path=self.local_path, **client_kwargs)
        except Exception as e2:
            if not silent_fallback:
                logger.warning(
                    f"[VectorStore] 本地持久化也失败 ({self.local_path}): {e2}，"
                    "回退到内存模式"
                )
            return QdrantClient(location=":memory:", **client_kwargs)

    def _ensure_collection(self):
        """确保 collection 存在"""
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            if self.collection_name not in names:
                logger.info(f"[VectorStore] 创建 collection: {self.collection_name}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    ),
                )
        except Exception as e:
            logger.error(f"[VectorStore] 初始化 collection 失败: {e}")
            raise

    def upsert(self, docs: List[VectorDocument]) -> None:
        """批量写入/更新文档"""
        if not docs:
            return

        texts = [d.content for d in docs]
        embeddings = self.embedding_service.encode(texts)

        points = []
        for doc, embedding in zip(docs, embeddings):
            doc.embedding = embedding
            points.append(
                PointStruct(
                    id=_normalize_point_id(doc.id),
                    vector=embedding,
                    payload={
                        "doc_type": doc.doc_type,
                        "session_id": doc.session_id,
                        "task_id": doc.task_id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"[VectorStore] 写入 {len(points)} 条向量文档")

    def search(
        self,
        query: str,
        doc_type: Optional[str] = None,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """文本检索"""
        vector = self.embedding_service.encode_single(query)
        return self.search_by_vector(vector, doc_type, top_k, session_id)

    def search_by_vector(
        self,
        vector: List[float],
        doc_type: Optional[str] = None,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """向量检索"""
        filters = []
        if doc_type:
            filters.append(
                FieldCondition(key="doc_type", match=MatchValue(value=doc_type))
            )
        if session_id:
            filters.append(
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            )

        from qdrant_client.models import Filter as QdrantFilter

        search_filter = QdrantFilter(must=filters) if filters else None

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=search_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": r.id,
                "score": r.score,
                "doc_type": r.payload.get("doc_type"),
                "session_id": r.payload.get("session_id"),
                "task_id": r.payload.get("task_id"),
                "content": r.payload.get("content"),
                "metadata": r.payload.get("metadata", {}),
            }
            for r in response.points
        ]

    def delete_by_session(self, session_id: str) -> None:
        """按会话删除文档"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
            ),
        )
        logger.info(f"[VectorStore] 删除 session {session_id} 的向量文档")

    def count(self) -> int:
        """获取文档总数"""
        result = self.client.count(collection_name=self.collection_name)
        return result.count


# 进程级单例缓存，避免每次实例化都重新连接 Qdrant/重新加载嵌入模型
_vector_store_cache: Dict[tuple, "VectorStore"] = {}
_vector_store_lock = threading.Lock()


def get_vector_store(
    url: Optional[str] = None,
    collection_name: Optional[str] = None,
    vector_size: int = VectorStore.DEFAULT_VECTOR_SIZE,
    embedding_service: Optional[EmbeddingService] = None,
) -> Optional[VectorStore]:
    """获取进程级单例 VectorStore。

    - 如果 ``VECTOR_STORE_ENABLED=false``，直接返回 ``None``。
    - 同一 (url, collection_name) 只初始化一次；远程 Qdrant 不可用时优先回退到
      本地持久化（``data/qdrant_storage``），最后才用 ``:memory:``。
    - 嵌入模型也只加载一次，避免 HuggingFace 重复探测。
    """
    if not settings.vector_store_enabled:
        return None

    url = url or settings.qdrant_url
    collection_name = collection_name or settings.qdrant_collection
    key = (url, collection_name)

    with _vector_store_lock:
        store = _vector_store_cache.get(key)
        if store is not None:
            return store

        store = VectorStore(
            url=url,
            collection_name=collection_name,
            vector_size=vector_size,
            embedding_service=embedding_service,
            silent_fallback=True,
        )
        # 如果最终落到内存模式且配置允许，显式提示一次
        if store.url == ":memory:" and getattr(settings, "qdrant_fallback_to_local", True):
            logger.warning(
                "[VectorStore] 当前使用内存模式，RAG 数据不会跨进程持久化；"
                "如需要持久化，请启动远程 Qdrant 或设置 QDRANT_URL 为本地路径"
            )

        _vector_store_cache[key] = store
        return store
