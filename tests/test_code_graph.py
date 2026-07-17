"""Unit tests for the code graph static analyzer."""

import pytest

from src.core.code_graph import CodeGraphAnalyzer


class TestCodeGraphAnalyzer:
    def test_empty_artifacts(self):
        result = CodeGraphAnalyzer.analyze([])
        assert result == {
            "avg_cyclomatic_complexity": 0.0,
            "max_cyclomatic_complexity": 0,
            "total_functions": 0,
            "total_classes": 0,
            "total_imports": 0,
            "unused_import_count": 0,
            "import_cycle_count": 0,
            "test_function_count": 0,
            "code_function_count": 0,
            "test_to_code_function_ratio": 0.0,
        }

    def test_normal_code_metrics(self):
        artifacts = [
            {
                "file_path": "src/app.py",
                "content": """
import os
import json

class Helper:
    pass

def process(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                continue
    return x

def simple(y):
    return y
""",
            }
        ]
        result = CodeGraphAnalyzer.analyze(artifacts)
        assert result["total_functions"] == 2
        assert result["total_classes"] == 1
        assert result["total_imports"] == 2
        # process: 1 base + if + for + if = 4
        # simple: 1
        assert result["max_cyclomatic_complexity"] == 4
        assert result["avg_cyclomatic_complexity"] == 2.5
        assert result["test_function_count"] == 0
        assert result["code_function_count"] == 2

    def test_unused_imports(self):
        artifacts = [
            {
                "file_path": "src/app.py",
                "content": """
import os
import json

def process(x):
    return os.path.join(x)
""",
            }
        ]
        result = CodeGraphAnalyzer.analyze(artifacts)
        assert result["total_imports"] == 2
        assert result["unused_import_count"] == 1

    def test_import_cycle(self):
        artifacts = [
            {
                "file_path": "src/a.py",
                "content": "from src import b\ndef foo(): pass",
            },
            {
                "file_path": "src/b.py",
                "content": "from src import a\ndef bar(): pass",
            },
        ]
        result = CodeGraphAnalyzer.analyze(artifacts)
        assert result["import_cycle_count"] == 1

    def test_no_cycle_for_external_imports(self):
        artifacts = [
            {
                "file_path": "src/a.py",
                "content": "import os\nimport sys\ndef foo(): pass",
            },
        ]
        result = CodeGraphAnalyzer.analyze(artifacts)
        assert result["import_cycle_count"] == 0

    def test_test_file_classification(self):
        artifacts = [
            {
                "file_path": "tests/test_app.py",
                "content": """
def test_foo():
    assert True

def helper():
    pass
""",
            },
            {
                "file_path": "src/app.py",
                "content": """
def production():
    pass
""",
            },
        ]
        result = CodeGraphAnalyzer.analyze(artifacts)
        assert result["test_function_count"] == 1
        assert result["code_function_count"] == 1
        assert result["test_to_code_function_ratio"] == 1.0

    def test_syntax_error_robustness(self):
        artifacts = [
            {
                "file_path": "src/broken.py",
                "content": "def foo(\n    pass",
            },
            {
                "file_path": "src/good.py",
                "content": "def bar(): pass",
            },
        ]
        result = CodeGraphAnalyzer.analyze(artifacts)
        assert result["total_functions"] == 1
        assert result["code_function_count"] == 1

    def test_skips_non_python_files(self):
        artifacts = [
            {
                "file_path": "README.md",
                "content": "# Project\n\nSome docs",
            },
            {
                "file_path": "src/app.py",
                "content": "def foo(): pass",
            },
        ]
        result = CodeGraphAnalyzer.analyze(artifacts)
        assert result["total_functions"] == 1
