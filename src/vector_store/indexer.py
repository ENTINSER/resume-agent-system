"""向量索引器

将任务结果索引到 Qdrant，供后续 RAG 检索使用。
Phase 5 增强：
- 确定性 ID，避免重复索引产生重复点
- 代码/文档分片（chunk），长文件不再截断
- 支持 README/requirements/测试文件等更多类型
- 索引测试成功/失败记录
"""

import hashlib
import json
from typing import Any, Dict, List, Tuple

from src.core.logger import logger
from src.vector_store.models import VectorDocument
from src.vector_store.vector_store import get_vector_store


# 文本分片配置
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def _doc_id(seed: str) -> str:
    """生成确定性 UUID 格式的文档 ID"""
    hex_str = hashlib.md5(seed.encode("utf-8")).hexdigest()
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"


def _chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[Tuple[int, str]]:
    """滑动窗口分片，返回 (chunk_index, chunk_text) 列表"""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [(0, text)]

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append((idx, text[start:end]))
        idx += 1
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _code_file_doc_type(file_path: str) -> str:
    """根据扩展名判断代码文档类型"""
    lower = file_path.lower()
    if lower.endswith(".py"):
        return "code"
    if lower.endswith((".md", "readme", "requirements.txt", ".toml", ".yaml", ".yml", ".json")):
        return "project_summary"
    if "test" in lower and lower.endswith(".py"):
        return "code"
    return "code"


def _index_artifact(
    artifact: Dict[str, Any],
    session_id: str,
    task_id: Any,
    project_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[VectorDocument]:
    """为单个代码产物生成向量文档"""
    file_path = artifact.get("file_path", "")
    content = artifact.get("content", "")
    if not content or not file_path:
        return []

    doc_type = _code_file_doc_type(file_path)
    docs: List[VectorDocument] = []

    for idx, chunk in _chunk_text(content, chunk_size, chunk_overlap):
        docs.append(VectorDocument(
            id=_doc_id(f"{session_id}|{doc_type}|{file_path}|{idx}"),
            doc_type=doc_type,
            session_id=session_id,
            task_id=task_id,
            content=f"文件: {file_path}\n\n{chunk}",
            metadata={
                "project_name": project_name,
                "file_path": file_path,
                "chunk_index": idx,
            },
        ))
    return docs


def _index_test_results(
    test_results: List[Dict[str, Any]],
    session_id: str,
    task_id: Any,
    project_name: str,
) -> List[VectorDocument]:
    """索引测试结果（成功/失败）作为 RAG 记忆"""
    docs: List[VectorDocument] = []
    for i, result in enumerate(test_results):
        passed = bool(result.get("passed", False))
        doc_type = "test_success" if passed else "test_failure"
        content = f"测试文件: {result.get('file', '')}\n通过: {passed}\n输出:\n{result.get('output', '')}"
        error = result.get("error", "")
        if error:
            content += f"\n错误:\n{error}"
        docs.append(VectorDocument(
            id=_doc_id(f"{session_id}|{doc_type}|{result.get('file', '')}|{i}"),
            doc_type=doc_type,
            session_id=session_id,
            task_id=task_id,
            content=content,
            metadata={
                "project_name": project_name,
                "file": result.get("file", ""),
                "passed": passed,
            },
        ))
    return docs


def index_task_result(state: Dict[str, Any]) -> None:
    """将任务结果索引到向量数据库"""
    store = get_vector_store()
    if store is None:
        logger.info("[Indexer] VECTOR_STORE_ENABLED=false，跳过索引")
        return

    session_id = state.get("session_id", "")
    task_id = state.get("task_id")
    requirements = state.get("requirements", "")
    project = state.get("project", {}) or {}
    evaluation = state.get("evaluation_result", {}) or {}
    code_artifacts = state.get("code_artifacts", []) or []
    test_results = state.get("test_results", []) or []

    project_name = project.get("name", "")
    docs: List[VectorDocument] = []

    # 1. 需求文档
    if requirements:
        docs.append(VectorDocument(
            id=_doc_id(f"{session_id}|requirement|"),
            doc_type="requirement",
            session_id=session_id,
            task_id=task_id,
            content=requirements,
            metadata={"project_name": project_name},
        ))

    # 2. 项目摘要
    if project_name:
        tech_stack = ", ".join(project.get("tech_stack", []))
        milestones = "\n".join(project.get("milestones", []))
        summary = (
            f"项目: {project_name}\n"
            f"技术栈: {tech_stack}\n"
            f"里程碑:\n{milestones}\n"
            f"需求: {requirements[:500]}"
        )
        docs.append(VectorDocument(
            id=_doc_id(f"{session_id}|project_summary|"),
            doc_type="project_summary",
            session_id=session_id,
            task_id=task_id,
            content=summary,
            metadata={
                "project_name": project_name,
                "tech_stack": project.get("tech_stack", []),
                "passed": evaluation.get("passed", False),
                "overall_score": evaluation.get("overall_score", 0),
            },
        ))

    # 3. 代码与文档分片索引
    for artifact in code_artifacts:
        docs.extend(_index_artifact(artifact, session_id, task_id, project_name))

    # 4. 评估报告
    if evaluation:
        eval_text = (
            f"项目: {project_name}\n"
            f"评分: {evaluation.get('overall_score', 0)}\n"
            f"通过: {evaluation.get('passed', False)}\n"
        )
        issues = evaluation.get("issues", []) or []
        if issues:
            eval_text += "问题:\n" + "\n".join(
                f"- {i.get('severity', '')}: {i.get('description', '')}" for i in issues[:5]
            )
        recommendations = evaluation.get("recommendations", []) or []
        if recommendations:
            eval_text += "\n建议:\n" + "\n".join(
                f"- {r.get('priority', '')}: {r.get('description', '')}" for r in recommendations[:5]
            )
        docs.append(VectorDocument(
            id=_doc_id(f"{session_id}|evaluation|"),
            doc_type="evaluation",
            session_id=session_id,
            task_id=task_id,
            content=eval_text,
            metadata={
                "project_name": project_name,
                "overall_score": evaluation.get("overall_score", 0),
                "passed": evaluation.get("passed", False),
            },
        ))

    # 5. 测试结果
    docs.extend(_index_test_results(test_results, session_id, task_id, project_name))

    if docs:
        try:
            store.upsert(docs)
            logger.info(f"[Indexer] 已索引 {len(docs)} 条向量文档")
        except Exception as e:
            logger.warning(f"[Indexer] 向量索引失败: {e}")
    else:
        logger.info("[Indexer] 无内容可索引")
