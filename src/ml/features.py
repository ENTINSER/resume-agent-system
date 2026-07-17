"""特征平台 - 从 state 提取标准化、版本化的 ML 特征

FEATURE_VERSION 用于控制特征模式，确保旧模型加载对应版本的特征。
"""

from typing import Dict, Any, List, Optional


FEATURE_VERSION = "v2"


class FeatureExtractor:
    """评估特征提取器"""

    @staticmethod
    def extract(state: Dict[str, Any]) -> Dict[str, Any]:
        """从任务 state 提取特征字典

        返回的特征可直接用于模型训练和推理。
        """
        static_metrics = state.get("static_metrics", {}) or {}
        test_results = state.get("test_results", []) or []
        project = state.get("project", {}) or {}

        # 测试信号
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results if r.get("passed", False))
        test_pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        all_tests_passed = total_tests > 0 and passed_tests == total_tests

        # 代码规模
        total_files = static_metrics.get("total_files", 0)
        python_files = static_metrics.get("python_files", 0)
        code_lines = max(static_metrics.get("code_lines", 0), 1)
        functions = max(static_metrics.get("functions", 1), 1)

        # 质量信号
        docstring_count = static_metrics.get("docstring_count", 0)
        type_hint_count = static_metrics.get("type_hint_count", 0)
        comment_lines = static_metrics.get("comment_lines", 0)
        syntax_errors = len(static_metrics.get("syntax_errors", []) or [])

        docstring_ratio = min(docstring_count / functions, 1.0)
        type_hint_ratio = min(type_hint_count / functions, 1.0)
        comment_ratio = min(comment_lines / code_lines, 1.0)

        # 结构信号
        has_readme = 1 if static_metrics.get("has_readme", False) else 0
        has_requirements = 1 if static_metrics.get("has_requirements", False) else 0
        has_tests = 1 if static_metrics.get("has_tests", False) else 0

        # 项目信号
        tech_stack = project.get("tech_stack", []) or []
        milestones = project.get("milestones", []) or []
        tech_stack_count = len(tech_stack)
        milestone_count = len(milestones)

        features = {
            "feature_version": FEATURE_VERSION,
            "total_files": total_files,
            "python_files": python_files,
            "code_lines": code_lines,
            "functions": functions,
            "classes": static_metrics.get("classes", 0),
            "imports": static_metrics.get("imports", 0),
            "test_functions": static_metrics.get("test_functions", 0),
            "docstring_count": docstring_count,
            "type_hint_count": type_hint_count,
            "comment_lines": comment_lines,
            "syntax_errors": syntax_errors,
            "docstring_ratio": round(docstring_ratio, 4),
            "type_hint_ratio": round(type_hint_ratio, 4),
            "comment_ratio": round(comment_ratio, 4),
            "has_readme": has_readme,
            "has_requirements": has_requirements,
            "has_tests": has_tests,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "test_pass_rate": round(test_pass_rate, 4),
            "all_tests_passed": 1 if all_tests_passed else 0,
            "tech_stack_count": tech_stack_count,
            "milestone_count": milestone_count,
            "graph_avg_cyclomatic_complexity": static_metrics.get("graph_avg_cyclomatic_complexity", 0.0),
            "graph_max_cyclomatic_complexity": static_metrics.get("graph_max_cyclomatic_complexity", 0),
            "graph_unused_import_count": static_metrics.get("graph_unused_import_count", 0),
            "graph_import_cycle_count": static_metrics.get("graph_import_cycle_count", 0),
            "graph_test_to_code_function_ratio": static_metrics.get("graph_test_to_code_function_ratio", 0.0),
        }
        return features

    @staticmethod
    def to_vector(features: Dict[str, Any]) -> List[float]:
        """把特征字典转换为模型输入向量

        保持字段顺序稳定，与训练时一致。
        """
        numeric_keys = [
            "total_files",
            "python_files",
            "code_lines",
            "functions",
            "classes",
            "imports",
            "test_functions",
            "docstring_count",
            "type_hint_count",
            "comment_lines",
            "syntax_errors",
            "docstring_ratio",
            "type_hint_ratio",
            "comment_ratio",
            "has_readme",
            "has_requirements",
            "has_tests",
            "total_tests",
            "passed_tests",
            "test_pass_rate",
            "all_tests_passed",
            "tech_stack_count",
            "milestone_count",
            "graph_avg_cyclomatic_complexity",
            "graph_max_cyclomatic_complexity",
            "graph_unused_import_count",
            "graph_import_cycle_count",
            "graph_test_to_code_function_ratio",
        ]
        return [float(features.get(k, 0.0)) for k in numeric_keys]

    @staticmethod
    def vector_columns() -> List[str]:
        """返回特征向量的列名顺序"""
        return [
            "total_files",
            "python_files",
            "code_lines",
            "functions",
            "classes",
            "imports",
            "test_functions",
            "docstring_count",
            "type_hint_count",
            "comment_lines",
            "syntax_errors",
            "docstring_ratio",
            "type_hint_ratio",
            "comment_ratio",
            "has_readme",
            "has_requirements",
            "has_tests",
            "total_tests",
            "passed_tests",
            "test_pass_rate",
            "all_tests_passed",
            "tech_stack_count",
            "milestone_count",
            "graph_avg_cyclomatic_complexity",
            "graph_max_cyclomatic_complexity",
            "graph_unused_import_count",
            "graph_import_cycle_count",
            "graph_test_to_code_function_ratio",
        ]
