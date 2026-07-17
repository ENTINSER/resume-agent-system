"""ML 特征提取单元测试"""

import pytest

from src.ml.features import FeatureExtractor


class TestFeatureExtractor:
    def test_extract_basic(self):
        state = {
            "static_metrics": {
                "total_files": 5,
                "python_files": 4,
                "code_lines": 120,
                "functions": 6,
                "classes": 1,
                "imports": 8,
                "test_functions": 3,
                "docstring_count": 4,
                "type_hint_count": 3,
                "comment_lines": 10,
                "syntax_errors": [],
                "has_readme": True,
                "has_requirements": True,
                "has_tests": True,
            },
            "test_results": [
                {"passed": True},
                {"passed": True},
                {"passed": False},
            ],
            "project": {
                "tech_stack": ["pytest", "fastapi"],
                "milestones": ["setup", "core", "eval"],
            },
        }
        features = FeatureExtractor.extract(state)
        assert features["feature_version"] == "v2"
        assert features["total_files"] == 5
        assert abs(features["test_pass_rate"] - 2 / 3) < 0.0001
        assert features["has_readme"] == 1
        assert features["tech_stack_count"] == 2
        assert features["milestone_count"] == 3

    def test_to_vector(self):
        state = {
            "static_metrics": {
                "total_files": 1,
                "python_files": 1,
                "code_lines": 10,
                "functions": 1,
                "classes": 0,
                "imports": 1,
                "test_functions": 0,
                "docstring_count": 0,
                "type_hint_count": 0,
                "comment_lines": 0,
                "syntax_errors": [],
                "has_readme": False,
                "has_requirements": False,
                "has_tests": False,
            },
            "test_results": [],
            "project": {},
        }
        features = FeatureExtractor.extract(state)
        vector = FeatureExtractor.to_vector(features)
        columns = FeatureExtractor.vector_columns()
        assert len(vector) == len(columns)
        assert all(isinstance(v, float) for v in vector)

    def test_graph_features_in_vector(self):
        state = {
            "static_metrics": {
                "total_files": 1,
                "python_files": 1,
                "code_lines": 10,
                "functions": 2,
                "classes": 0,
                "imports": 2,
                "test_functions": 1,
                "docstring_count": 0,
                "type_hint_count": 0,
                "comment_lines": 0,
                "syntax_errors": [],
                "has_readme": False,
                "has_requirements": False,
                "has_tests": False,
                "graph_avg_cyclomatic_complexity": 1.5,
                "graph_max_cyclomatic_complexity": 3,
                "graph_unused_import_count": 1,
                "graph_import_cycle_count": 2,
                "graph_test_to_code_function_ratio": 0.5,
            },
            "test_results": [],
            "project": {},
        }
        features = FeatureExtractor.extract(state)
        vector = FeatureExtractor.to_vector(features)
        columns = FeatureExtractor.vector_columns()
        assert columns[-5:] == [
            "graph_avg_cyclomatic_complexity",
            "graph_max_cyclomatic_complexity",
            "graph_unused_import_count",
            "graph_import_cycle_count",
            "graph_test_to_code_function_ratio",
        ]
        assert vector[-5:] == [1.5, 3.0, 1.0, 2.0, 0.5]
