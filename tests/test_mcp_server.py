"""MCP Server 工具函数单元测试"""

import os
import tempfile

import pytest

from src.mcp.server import (
    _resolve_path,
    _read_file_sync,
    _write_file_sync,
    _list_dir_sync,
    _is_dangerous_command,
)


class TestMCPServerHelpers:
    @pytest.fixture
    def workspace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_resolve_path_within_workspace(self, workspace):
        path = _resolve_path(workspace, "sub/file.txt")
        assert path.resolve() == (Path(workspace) / "sub" / "file.txt").resolve()

    def test_resolve_path_escape(self, workspace):
        with pytest.raises(ValueError):
            _resolve_path(workspace, "../etc/passwd")

    def test_read_write_list(self, workspace):
        result = _write_file_sync(workspace, "hello.txt", "world")
        assert "成功" in result

        content = _read_file_sync(workspace, "hello.txt")
        assert content == "world"

        listing = _list_dir_sync(workspace, ".")
        assert "FILE: hello.txt" in listing

    def test_read_missing_file(self, workspace):
        result = _read_file_sync(workspace, "missing.txt")
        assert "不存在" in result

    def test_is_dangerous_command(self):
        assert _is_dangerous_command("rm -rf /") is True
        assert _is_dangerous_command("ls -la") is False


from pathlib import Path
