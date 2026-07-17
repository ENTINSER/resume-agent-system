"""外部仓库种子

支持从本地目录或 git 仓库导入代码/文档作为 RAG 标杆案例。
"""

import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from src.core.logger import logger
from src.vector_store.models import VectorDocument
from src.vector_store.vector_store import get_vector_store
from src.vector_store.indexer import _chunk_text, _doc_id


DEFAULT_BLACKLIST = {
    ".git",
    ".github",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".idea",
    ".vscode",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}

DEFAULT_CODE_PATTERNS = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c", ".h"}
DEFAULT_DOC_PATTERNS = {".md", ".txt", ".rst", ".toml", ".yaml", ".yml", ".json"}


def _is_blacklisted(path_parts: List[str]) -> bool:
    for part in path_parts:
        name = Path(part).name
        if name in DEFAULT_BLACKLIST:
            return True
        if any(part.endswith(suffix) for suffix in (".egg-info",)):
            return True
    return False


def _is_text_file(file_path: Path) -> bool:
    """通过扩展名和大小粗略判断是否为可索引文本文件"""
    suffix = file_path.suffix.lower()
    if suffix in DEFAULT_CODE_PATTERNS | DEFAULT_DOC_PATTERNS:
        return True
    # 无扩展名文件可能是脚本
    if not suffix and file_path.stat().st_size < 1024 * 1024:
        return True
    return False


def _doc_type_for_file(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in DEFAULT_CODE_PATTERNS:
        return "external_code"
    return "external_doc"


class ExternalRepoSeeder:
    """外部仓库种子器"""

    def __init__(
        self,
        source: str,
        ref: Optional[str] = None,
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
        max_file_size: int = 1024 * 1024,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.source = source
        self.ref = ref or "main"
        self.tags = tags or []
        self.name = name or self._derive_name()
        self.max_file_size = max_file_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _derive_name(self) -> str:
        if os.path.exists(self.source):
            return Path(self.source).name
        parsed = urlparse(self.source)
        path = parsed.path.strip("/")
        return path.split("/")[-1].replace(".git", "") if path else "external_repo"

    def _prepare_dir(self) -> str:
        if os.path.exists(self.source):
            return self.source

        tmpdir = tempfile.mkdtemp(prefix="rag_seed_")
        refs_to_try = []
        if self.ref:
            refs_to_try.append(self.ref)
        refs_to_try.extend(["main", "master"])
        refs_to_try.append(None)

        last_error = ""
        for ref in refs_to_try:
            clone_cmd = ["git", "clone", "--depth", "1"]
            if ref:
                clone_cmd.extend(["--branch", ref])
            clone_cmd.extend([self.source, tmpdir])
            logger.info(f"[ExternalRepoSeeder] 克隆仓库: {' '.join(clone_cmd)}")
            result = subprocess.run(clone_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return tmpdir
            last_error = result.stderr.strip() or result.stdout.strip()
            logger.warning(f"[ExternalRepoSeeder] 分支 {ref!r} 克隆失败: {last_error[:200]}")
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
                tmpdir = tempfile.mkdtemp(prefix="rag_seed_")
            except Exception:
                pass

        raise RuntimeError(
            f"[ExternalRepoSeeder] 无法克隆仓库 {self.source}，最后错误: {last_error[:500]}"
        )

    def seed(self) -> Dict[str, Any]:
        """执行索引，返回统计信息"""
        store = get_vector_store()
        if store is None:
            return {"indexed": 0, "error": "VECTOR_STORE_ENABLED=false"}

        repo_dir = self._prepare_dir()
        docs: List[VectorDocument] = []
        indexed_files = 0

        for root, _dirs, files in os.walk(repo_dir):
            rel_root = os.path.relpath(root, repo_dir)
            if rel_root == ".":
                rel_root = ""
            parts = rel_root.split(os.sep) if rel_root else []
            if _is_blacklisted(parts):
                continue

            for filename in files:
                file_path = Path(root) / filename
                if not _is_text_file(file_path):
                    continue
                if file_path.stat().st_size > self.max_file_size:
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    logger.warning(f"[ExternalRepoSeeder] 读取失败 {file_path}: {e}")
                    continue

                rel_path = os.path.join(rel_root, filename) if rel_root else filename
                doc_type = _doc_type_for_file(file_path)
                for idx, chunk in _chunk_text(content, self.chunk_size, self.chunk_overlap):
                    docs.append(VectorDocument(
                        id=_doc_id(f"external|{self.source}|{self.ref}|{rel_path}|{idx}"),
                        doc_type=doc_type,
                        session_id="",
                        task_id=None,
                        content=f"仓库: {self.name}\n文件: {rel_path}\n\n{chunk}",
                        metadata={
                            "source": "external_repo",
                            "repo_url": self.source,
                            "repo_ref": self.ref,
                            "repo_name": self.name,
                            "file_path": rel_path,
                            "tags": self.tags,
                            "chunk_index": idx,
                        },
                    ))
                indexed_files += 1

        if docs:
            store.upsert(docs)

        logger.info(
            f"[ExternalRepoSeeder] 索引完成: {self.name}, "
            f"文件数 {indexed_files}, 文档数 {len(docs)}"
        )
        return {
            "indexed": len(docs),
            "files": indexed_files,
            "repo": self.name,
            "repo_dir": repo_dir,
        }
