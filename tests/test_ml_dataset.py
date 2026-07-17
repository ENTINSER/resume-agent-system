"""ML 样本仓库单元测试"""

import tempfile

import pytest

from src.ml.dataset import EvaluationDataset


class TestEvaluationDataset:
    @pytest.fixture
    def dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield EvaluationDataset(dataset_dir=tmpdir, version="v1")

    def test_append_and_load(self, dataset):
        state = {
            "session_id": "s1",
            "task_id": "t1",
            "static_metrics": {
                "total_files": 2,
                "python_files": 1,
                "code_lines": 50,
                "functions": 2,
                "classes": 0,
                "imports": 2,
                "test_functions": 1,
                "docstring_count": 1,
                "type_hint_count": 1,
                "comment_lines": 2,
                "syntax_errors": [],
                "has_readme": True,
                "has_requirements": True,
                "has_tests": True,
            },
            "test_results": [{"passed": True}],
            "project": {},
        }
        result = {"overall_score": 85, "passed": True, "source": "llm"}

        assert dataset.append(state, result) is True
        samples = dataset.load()
        assert len(samples) == 1
        assert samples[0]["labels"]["overall_score"] == 85

    def test_deduplicate(self, dataset):
        state = {
            "session_id": "s1",
            "task_id": "t1",
            "static_metrics": {"total_files": 1, "python_files": 1, "code_lines": 10, "functions": 1, "classes": 0, "imports": 1, "test_functions": 0, "docstring_count": 0, "type_hint_count": 0, "comment_lines": 0, "syntax_errors": [], "has_readme": False, "has_requirements": False, "has_tests": False},
            "test_results": [],
            "project": {},
        }
        result = {"overall_score": 60, "passed": False}
        dataset.append(state, result)
        assert dataset.append(state, result) is False

    def test_get_xy(self, dataset):
        state = {
            "session_id": "s1",
            "task_id": "t1",
            "static_metrics": {"total_files": 1, "python_files": 1, "code_lines": 10, "functions": 1, "classes": 0, "imports": 1, "test_functions": 0, "docstring_count": 0, "type_hint_count": 0, "comment_lines": 0, "syntax_errors": [], "has_readme": False, "has_requirements": False, "has_tests": False},
            "test_results": [],
            "project": {},
        }
        dataset.append(state, {"overall_score": 80, "passed": True})
        X, y_score, y_passed = dataset.get_xy()
        assert len(X) == 1
        assert y_score[0] == 80
        assert y_passed[0] == 1
