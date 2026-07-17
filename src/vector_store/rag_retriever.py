"""RAG 检索器（向后兼容入口）

此模块保留早期 ``src.vector_store.rag_retriever.RAGRetriever`` 的导入路径，
实际实现委托给 ``src.vector_store.retrieval.RAGRetriever``，并补充
``format_context`` 等历史工具方法。
"""

from typing import Any, Dict, List, Optional

from src.vector_store.retrieval import RAGRetriever as _BaseRetriever
from src.vector_store.retrieval import validate_doc_type


class RAGRetriever(_BaseRetriever):
    """RAG 检索器（兼容旧接口）"""

    def __init__(self, vector_store: Optional[Any] = None):
        super().__init__(vector_store=vector_store)

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """将检索结果格式化为 prompt 可用的上下文"""
        if not results:
            return ""

        lines = ["### 相似历史案例（按相似度排序）"]
        for i, r in enumerate(results, 1):
            score = r.get("score", 0)
            content = r.get("content", "")[:800]
            meta = r.get("metadata", {})
            project_name = meta.get("project_name", "未知项目")
            lines.append(
                f"\n案例 {i}（相似度: {score:.3f}，项目: {project_name}）:\n{content}\n"
            )
        return "\n".join(lines)


__all__ = ["RAGRetriever", "validate_doc_type"]
