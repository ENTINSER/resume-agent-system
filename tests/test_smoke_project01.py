"""项目 01 冒烟测试

验证 projects/01-langgraph-multi-agent 在当前 Python 环境下
能完整运行其测试套件，达到 README 声明的可用质量标准。
此测试不调用真实 LLM，属于无成本冒烟。
"""

import os
import subprocess
import sys

import pytest


PROJECT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "projects", "01-langgraph-multi-agent"
)


class TestProject01Smoke:
    """项目 01 本地冒烟测试"""

    def test_project_01_tests_pass_locally(self):
        project_dir = os.path.abspath(PROJECT_DIR)
        assert os.path.exists(project_dir), f"项目目录不存在: {project_dir}"

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, (
            f"项目 01 测试未通过:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
