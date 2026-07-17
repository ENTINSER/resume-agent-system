"""SFT 多角色数据采集器单元测试"""

import pytest

from src.sft.collectors import DCollector, BCollector, ECollector, CCollector, SFTCollectorPipeline


class TestCollectors:
    def test_d_collector(self):
        state = {
            "session_id": "s1",
            "task_id": "t1",
            "requirements": "开发一个 Python 计算器",
            "project": {
                "name": "Calculator",
                "tech_stack": ["pytest"],
                "milestones": ["setup", "core"],
            },
            "evaluation_result": {"overall_score": 85, "passed": True},
        }
        collector = DCollector()
        assert collector.can_collect(state) is True
        samples = collector.collect(state)
        assert len(samples) == 1
        assert samples[0].role == "D"
        assert "Calculator" in samples[0].output

    def test_b_collector(self):
        state = {
            "session_id": "s1",
            "task_id": "t1",
            "requirements": "开发一个 Python 计算器",
            "project": {"name": "Calculator"},
            "code_artifacts": [
                {"file_path": "main.py", "content": "def add(a, b): return a + b"},
                {"file_path": "tests/test_main.py", "content": "def test_add(): assert add(1,2)==3"},
            ],
            "evaluation_result": {"overall_score": 85, "passed": True},
        }
        collector = BCollector()
        assert collector.can_collect(state) is True
        samples = collector.collect(state)
        assert len(samples) == 1
        assert samples[0].role == "B"
        assert "=== FILE: main.py ===" in samples[0].output

    def test_e_collector(self):
        state = {
            "session_id": "s1",
            "task_id": "t1",
            "requirements": "开发一个 Python 计算器",
            "code_artifacts": [{"file_path": "main.py", "content": "def add(a, b): return a + b"}],
            "test_results": [{"file": "tests/test_main.py", "passed": True}],
            "static_metrics": {"total_files": 2, "code_lines": 10},
            "evaluation_result": {"overall_score": 85, "passed": True},
        }
        collector = ECollector()
        assert collector.can_collect(state) is True
        samples = collector.collect(state)
        assert len(samples) == 1
        assert samples[0].role == "E"

    def test_c_collector(self):
        state = {
            "session_id": "s1",
            "task_id": "t1",
            "requirements": "开发一个 Python 计算器",
            "code_artifacts": [
                {"file_path": "main.py", "content": "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b", "language": "python"},
            ],
            "evaluation_result": {"overall_score": 85, "passed": True},
        }
        collector = CCollector()
        assert collector.can_collect(state) is True
        samples = collector.collect(state)
        assert len(samples) == 2
        assert all(s.role == "C" for s in samples)

    def test_pipeline(self):
        state = {
            "session_id": "s1",
            "task_id": "t1",
            "requirements": "开发一个 Python 计算器",
            "project": {"name": "Calculator"},
            "code_artifacts": [
                {"file_path": "main.py", "content": "def add(a, b): return a + b", "language": "python"},
            ],
            "test_results": [{"file": "tests/test_main.py", "passed": True}],
            "static_metrics": {"total_files": 2, "code_lines": 10},
            "evaluation_result": {"overall_score": 85, "passed": True},
        }
        pipeline = SFTCollectorPipeline(roles=["D", "B", "E", "C"])
        samples = pipeline.collect(state)
        roles = {s.role for s in samples}
        assert "D" in roles
        assert "B" in roles
        assert "E" in roles
        assert "C" in roles
