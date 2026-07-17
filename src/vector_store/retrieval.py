"""RAG 检索策略层

提供面向业务场景的检索方法：
- D 阶段：相似需求/项目
- B 阶段：相似代码
- E 阶段：相似评估/失败模式
- 混合检索（dense + keyword）、rerank、多跳检索
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import settings
from src.core.logger import logger
from src.vector_store.embedding_service import EmbeddingService
from src.vector_store.models import SUPPORTED_DOC_TYPES
from src.vector_store.vector_store import VectorStore, get_vector_store


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[\w\-]+\b", text.lower())


def _keyword_score(query: str, content: str) -> float:
    q_tokens = set(_tokenize(query))
    c_tokens = _tokenize(content)
    if not q_tokens or not c_tokens:
        return 0.0
    overlap = q_tokens & set(c_tokens)
    if not overlap:
        return 0.0
    # 简单 TF-IDF 近似：命中词频/内容长度
    tf = sum(c_tokens.count(t) for t in overlap)
    return len(overlap) / len(q_tokens) * (tf / len(c_tokens))


class RAGRetriever:
    """RAG 检索器"""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        rerank_model: Optional[str] = None,
    ):
        self.vector_store = vector_store or get_vector_store()
        self.embedding_service = embedding_service or EmbeddingService()
        self.rerank_model_name = rerank_model or getattr(settings, "rag_rerank_model", None)
        self._reranker = None

    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results:
            return []
        if self.rerank_model_name:
            try:
                if self._reranker is None:
                    from sentence_transformers import CrossEncoder
                    self._reranker = CrossEncoder(self.rerank_model_name)
                pairs = [(query, r.get("content", "")) for r in results]
                scores = self._reranker.predict(pairs)
                for r, score in zip(results, scores):
                    r["rerank_score"] = float(score)
                results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
                return results
            except Exception as e:
                logger.warning(f"[RAGRetriever] rerank 失败: {e}, 使用 dense+keyword 混合分")
        # fallback: keyword 加成
        for r in results:
            kw = _keyword_score(query, r.get("content", ""))
            r["keyword_score"] = kw
            r["hybrid_score"] = r.get("score", 0) * 0.7 + kw * 0.3
        results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        return results

    def _search(
        self,
        query: str,
        doc_type: Optional[str] = None,
        top_k: int = 5,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self.vector_store is None:
            return []
        # 混合检索先扩大召回面
        candidates = self.vector_store.search(
            query,
            doc_type=doc_type,
            top_k=max(top_k * 4, 20),
            session_id=session_id,
        )
        reranked = self._rerank(query, candidates)
        return reranked[:top_k]

    def find_similar_projects(self, requirements: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """D 阶段：检索相似历史需求/项目

        优先召回 ``project_summary``，不足时再补充 ``historical_doc`` /
        ``requirement``，保证项目级检索的相关性。
        """
        primary = self._search(requirements, doc_type="project_summary", top_k=top_k)
        if len(primary) >= top_k:
            return primary

        seen = {r.get("id") for r in primary}
        for dtype in ("historical_doc", "requirement"):
            extra = self._search(requirements, doc_type=dtype, top_k=top_k)
            for r in extra:
                if r.get("id") not in seen:
                    primary.append(r)
                    seen.add(r.get("id"))
            if len(primary) >= top_k:
                break
        return self._rerank(requirements, primary)[:top_k]

    def find_similar_code(self, requirements: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """B 阶段：检索相似代码片段"""
        code_types = {"code", "historical_code", "external_code"}
        all_results: List[Dict[str, Any]] = []
        for dtype in code_types:
            all_results.extend(self._search(requirements, doc_type=dtype, top_k=top_k))
        # 合并后再 rerank
        seen = set()
        unique = []
        for r in all_results:
            rid = r.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            unique.append(r)
        return self._rerank(requirements, unique)[:top_k]

    def find_similar_evaluations(self, requirements: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """E 阶段：检索相似评估与测试失败模式"""
        eval_types = {"evaluation", "test_failure", "test_success"}
        all_results: List[Dict[str, Any]] = []
        for dtype in eval_types:
            all_results.extend(self._search(requirements, doc_type=dtype, top_k=top_k))
        return self._rerank(requirements, all_results)[:top_k]

    def find_failure_patterns(self, error_message: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """根据测试错误信息检索失败模式"""
        return self._search(error_message, doc_type="test_failure", top_k=top_k)

    def multi_step_retrieve(
        self,
        requirements: str,
        top_k_per_step: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """多跳检索：需求 -> 相关项目 -> 相关代码/评估"""
        similar_projects = self.find_similar_projects(requirements, top_k=top_k_per_step)
        session_ids = [
            r["session_id"]
            for r in similar_projects
            if r.get("session_id")
        ]

        code_results: List[Dict[str, Any]] = []
        eval_results: List[Dict[str, Any]] = []
        for sid in session_ids:
            code_results.extend(
                self._search(requirements, doc_type="historical_code", top_k=top_k_per_step, session_id=sid)
            )
            eval_results.extend(
                self._search(requirements, doc_type="evaluation", top_k=top_k_per_step, session_id=sid)
            )

        return {
            "projects": similar_projects,
            "code": self._rerank(requirements, code_results)[:top_k_per_step * len(session_ids)],
            "evaluations": self._rerank(requirements, eval_results)[:top_k_per_step * len(session_ids)],
        }

    def retrieve_for_generation(
        self,
        requirements: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """为代码生成组装上下文：需求 + 代码 + 失败模式"""
        return {
            "similar_projects": self.find_similar_projects(requirements, top_k=top_k),
            "similar_code": self.find_similar_code(requirements, top_k=top_k),
            "failure_patterns": self.find_failure_patterns(requirements, top_k=top_k),
        }


def validate_doc_type(doc_type: Optional[str]) -> Optional[str]:
    """校验 doc_type，未知类型返回 None 供调用方显式处理"""
    if doc_type is None:
        return None
    if doc_type in SUPPORTED_DOC_TYPES:
        return doc_type
    logger.warning(f"[RAGRetriever] 未知 doc_type: {doc_type}，将忽略类型过滤")
    return None
