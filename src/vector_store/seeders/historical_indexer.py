"""历史项目/会话索引

从 `SessionStore` 与 `projects/` 目录中，把已完成的会话重新索引到 RAG，
为后续生成提供相似需求、相似代码、测试失败模式等上下文。
"""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger
from src.core.persistence import get_session_store
from src.vector_store.indexer import _chunk_text, _doc_id
from src.vector_store.models import VectorDocument
from src.vector_store.vector_store import get_vector_store


# 与 indexer 保持一致的文本文件判定
_TEXT_SUFFIXES = {
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h",
    ".md", ".txt", ".rst", ".toml", ".yaml", ".yml", ".json",
}
_BLACKLIST = {
    ".git", "__pycache__", ".pytest_cache", "node_modules",
    ".venv", "venv", ".idea", ".vscode", "dist", "build",
    ".eggs", ".egg-info", ".tox",
}


def _is_blacklisted(parts: List[str]) -> bool:
    return any(Path(p).name in _BLACKLIST for p in parts)


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES


class HistoricalIndexer:
    """历史项目索引器"""

    def __init__(
        self,
        projects_dir: Optional[str] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.projects_dir = Path(projects_dir or settings.projects_dir or "projects")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store = get_session_store()

    def index_sessions(self) -> Dict[str, int]:
        """把数据库中 `passed=1` 的完成会话索引为标杆案例"""
        vector_store = get_vector_store()
        if vector_store is None:
            return {"indexed": 0}

        sessions = self.store.list_sessions(limit=1000)
        completed = [s for s in sessions if s.get("status") == "complete" and s.get("passed")]

        docs: List[VectorDocument] = []
        for session in completed:
            sid = session["session_id"]
            requirements = session.get("requirements", "")
            project_name = session.get("project_name", "") or sid

            if requirements:
                docs.append(VectorDocument(
                    id=_doc_id(f"historical|{sid}|requirements"),
                    doc_type="historical_doc",
                    session_id=sid,
                    task_id=None,
                    content=f"历史项目需求\n项目名称: {project_name}\n\n{requirements}",
                    metadata={
                        "source": "historical_session",
                        "project_name": project_name,
                        "doc_subtype": "requirements",
                        "overall_score": session.get("overall_score", 0),
                        "status": session.get("status"),
                    },
                ))
                for idx, chunk in _chunk_text(requirements, self.chunk_size, self.chunk_overlap):
                    docs.append(VectorDocument(
                        id=_doc_id(f"historical|{sid}|requirements_chunk|{idx}"),
                        doc_type="historical_doc",
                        session_id=sid,
                        task_id=None,
                        content=f"历史项目需求片段\n项目名称: {project_name}\n\n{chunk}",
                        metadata={
                            "source": "historical_session",
                            "project_name": project_name,
                            "doc_subtype": "requirements_chunk",
                            "chunk_index": idx,
                        },
                    ))

            # 代码产物
            artifacts = self._load_artifacts(sid)
            for artifact in artifacts:
                file_path = artifact.get("file_path", "")
                content = artifact.get("content", "")
                if not content:
                    continue
                ext = Path(file_path).suffix.lower()
                doc_type = "historical_code" if ext in {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h"} else "historical_doc"
                for idx, chunk in _chunk_text(content, self.chunk_size, self.chunk_overlap):
                    docs.append(VectorDocument(
                        id=_doc_id(f"historical|{sid}|artifact|{file_path}|{idx}"),
                        doc_type=doc_type,
                        session_id=sid,
                        task_id=None,
                        content=f"历史项目代码\n项目: {project_name}\n文件: {file_path}\n\n{chunk}",
                        metadata={
                            "source": "historical_session",
                            "project_name": project_name,
                            "file_path": file_path,
                            "language": artifact.get("language", ""),
                            "chunk_index": idx,
                        },
                    ))

            # 评估摘要
            eval_rows = self._load_evaluations(sid)
            for ev in eval_rows:
                score = ev.get("overall_score", 0)
                content = (
                    f"历史项目评估\n项目: {project_name}\n"
                    f"总分: {score}\n"
                    f"代码质量: {ev.get('code_quality', '')}\n"
                    f"功能: {ev.get('functionality', '')}\n"
                    f"性能: {ev.get('performance', '')}\n"
                    f"文档: {ev.get('documentation', '')}"
                )
                docs.append(VectorDocument(
                    id=_doc_id(f"historical|{sid}|evaluation|{ev.get('iteration', 0)}"),
                    doc_type="evaluation",
                    session_id=sid,
                    task_id=None,
                    content=content,
                    metadata={
                        "source": "historical_session",
                        "project_name": project_name,
                        "doc_subtype": "evaluation",
                        "overall_score": score,
                        "passed": bool(ev.get("passed")),
                    },
                ))

            # 测试结果
            test_rows = self._load_tests(sid)
            for tr in test_rows:
                passed = bool(tr.get("passed"))
                doc_type = "test_success" if passed else "test_failure"
                content = (
                    f"历史项目测试{'通过' if passed else '失败'}\n"
                    f"项目: {project_name}\n文件: {tr.get('file', '')}\n"
                    f"输出:\n{tr.get('output', '')}\n错误:\n{tr.get('error', '')}"
                )
                docs.append(VectorDocument(
                    id=_doc_id(f"historical|{sid}|test|{tr.get('file', '')}|{tr.get('iteration', 0)}"),
                    doc_type=doc_type,
                    session_id=sid,
                    task_id=None,
                    content=content,
                    metadata={
                        "source": "historical_session",
                        "project_name": project_name,
                        "file": tr.get("file", ""),
                        "passed": passed,
                        "error": tr.get("error", ""),
                    },
                ))

        if docs:
            vector_store.upsert(docs)

        logger.info(f"[HistoricalIndexer] 会话索引完成: {len(docs)} 文档, {len(completed)} 会话")
        return {"indexed": len(docs), "sessions": len(completed)}

    def index_projects_dir(self) -> Dict[str, int]:
        """把 `projects/` 目录中未索引/已存在目录索引为历史代码"""
        vector_store = get_vector_store()
        if vector_store is None or not self.projects_dir.exists():
            return {"indexed": 0}

        docs: List[VectorDocument] = []
        indexed_dirs = 0

        for project_path in self.projects_dir.iterdir():
            if not project_path.is_dir():
                continue
            indexed_dirs += 1
            project_name = project_path.name
            for root, _dirs, files in os.walk(project_path):
                rel_root = os.path.relpath(root, project_path)
                parts = rel_root.split(os.sep) if rel_root != "." else []
                if _is_blacklisted(parts):
                    continue
                for filename in files:
                    file_path = Path(root) / filename
                    if not _is_text_file(file_path):
                        continue
                    try:
                        content = file_path.read_text(encoding="utf-8", errors="ignore")
                    except Exception as e:
                        logger.warning(f"[HistoricalIndexer] 读取失败 {file_path}: {e}")
                        continue
                    rel_file = os.path.join(rel_root, filename) if rel_root != "." else filename
                    doc_type = "historical_code" if file_path.suffix.lower() in {
                        ".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h"
                    } else "historical_doc"
                    for idx, chunk in _chunk_text(content, self.chunk_size, self.chunk_overlap):
                        docs.append(VectorDocument(
                            id=_doc_id(f"historical_dir|{project_name}|{rel_file}|{idx}"),
                            doc_type=doc_type,
                            session_id="",
                            task_id=None,
                            content=f"历史项目目录\n项目: {project_name}\n文件: {rel_file}\n\n{chunk}",
                            metadata={
                                "source": "historical_project_dir",
                                "project_name": project_name,
                                "file_path": rel_file,
                                "chunk_index": idx,
                            },
                        ))

        if docs:
            vector_store.upsert(docs)

        logger.info(f"[HistoricalIndexer] 项目目录索引完成: {len(docs)} 文档, {indexed_dirs} 目录")
        return {"indexed": len(docs), "project_dirs": indexed_dirs}

    def index_all(self) -> Dict[str, Any]:
        session_stats = self.index_sessions()
        project_stats = self.index_projects_dir()
        return {"sessions": session_stats, "projects": project_stats}

    def _load_artifacts(self, session_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT file_path, content, language FROM artifacts WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def _load_evaluations(self, session_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT iteration, overall_score, code_quality, functionality, performance, documentation, passed FROM evaluations WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def _load_tests(self, session_id: str) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.store.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT iteration, file, passed, output, error FROM test_runs WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
