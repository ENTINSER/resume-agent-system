"""save_project / 目标目录解析测试"""

import os
import tempfile

from src.core.B import save_project
from src.tasks.agent_task import _extract_target_dir


def test_extract_target_dir():
    assert _extract_target_dir("在 projects/test-calculator-v1 目录下实现...") == "test-calculator-v1"
    assert _extract_target_dir("在projects/demo-project 下...") == "demo-project"
    assert _extract_target_dir("实现一个计算器") is None


def test_save_project_uses_hint():
    state = {
        "project": {"id": "proj-1234abcd", "name": "CalcCLI"},
        "code_artifacts": [
            {"file_path": "main.py", "content": "print('hello')"},
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        saved = save_project(state, tmpdir, project_dir_hint="test-calculator-v1")
        assert saved.endswith("test-calculator-v1")
        assert os.path.exists(os.path.join(saved, "main.py"))


def test_save_project_fallback():
    state = {
        "project": {"id": "proj-1234abcd", "name": "CalcCLI"},
        "code_artifacts": [
            {"file_path": "main.py", "content": "print('hello')"},
        ],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        saved = save_project(state, tmpdir)
        assert "calccli-abcd" in saved
        assert os.path.exists(os.path.join(saved, "main.py"))
