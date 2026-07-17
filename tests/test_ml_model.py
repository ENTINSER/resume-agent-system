"""ML 模型训练与预测单元测试"""

import tempfile

import pytest

from src.ml.model import EvalModel, ModelTrainer
from src.ml.features import FeatureExtractor


class TestEvalModel:
    def test_fit_and_predict(self):
        # 构造简单训练数据：代码行数越多分越高
        X = []
        y_score = []
        y_passed = []
        for i in range(10):
            features = FeatureExtractor.extract({
                "static_metrics": {
                    "total_files": 1 + i,
                    "python_files": 1 + i,
                    "code_lines": 10 + i * 10,
                    "functions": 1 + i,
                    "classes": 0,
                    "imports": 1 + i,
                    "test_functions": 1 if i > 3 else 0,
                    "docstring_count": i,
                    "type_hint_count": i,
                    "comment_lines": i,
                    "syntax_errors": [],
                    "has_readme": i % 2,
                    "has_requirements": i % 2,
                    "has_tests": 1 if i > 3 else 0,
                },
                "test_results": [{"passed": True}] if i > 3 else [],
                "project": {"tech_stack": ["pytest"], "milestones": ["setup"]},
            })
            X.append(FeatureExtractor.to_vector(features))
            y_score.append(50 + i * 5)
            y_passed.append(1 if i > 3 else 0)

        model = EvalModel()
        model.fit(X, y_score, y_passed)

        scores, passed = model.predict(X[:3])
        assert len(scores) == 3
        assert len(passed) == 3

        importances = model.feature_importances()
        assert "code_lines" in importances

    def test_save_and_load(self):
        model = EvalModel()
        X = [[0.0] * len(FeatureExtractor.vector_columns())]
        y_score = [70.0]
        y_passed = [1]
        model.fit(X, y_score, y_passed)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            loaded = EvalModel.load(tmpdir)
            scores, passed = loaded.predict(X)
            assert len(scores) == 1


class TestModelTrainer:
    def test_train(self):
        from src.ml.dataset import EvaluationDataset

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = EvaluationDataset(dataset_dir=tmpdir, version="v1")
            for i in range(10):
                state = {
                    "session_id": f"s{i}",
                    "task_id": f"t{i}",
                    "static_metrics": {
                        "total_files": 1 + i,
                        "python_files": 1 + i,
                        "code_lines": 10 + i * 10,
                        "functions": 1 + i,
                        "classes": 0,
                        "imports": 1 + i,
                        "test_functions": 1 if i > 3 else 0,
                        "docstring_count": i,
                        "type_hint_count": i,
                        "comment_lines": i,
                        "syntax_errors": [],
                        "has_readme": i % 2,
                        "has_requirements": i % 2,
                        "has_tests": 1 if i > 3 else 0,
                    },
                    "test_results": [{"passed": True}] if i > 3 else [],
                    "project": {},
                }
                dataset.append(state, {"overall_score": 50 + i * 5, "passed": i > 3})

            with tempfile.TemporaryDirectory() as model_dir:
                trainer = ModelTrainer(model_dir=model_dir)
                report = trainer.train(dataset)
                assert "experiment_id" in report
                assert "metrics" in report
                assert "model_path" in report
