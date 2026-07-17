"""ML 训练 Celery Task 单元测试"""

import os
import tempfile

import pytest

from src.tasks.ml_training_task import train_ml_evaluator
from src.ml.dataset import EvaluationDataset


class TestMLTrainingTask:
    def test_skip_when_insufficient_samples(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("ML_EVAL_MIN_SAMPLES", "100")
            monkeypatch.setenv("ML_EVAL_AUTO_STAGE", "shadow")

            #  monkeypatch dataset default dir is hardcoded, so we directly test logic
            #  just run task and expect skip because count < 100
            result = train_ml_evaluator.run(min_samples=100, stage="shadow")
            assert result["trained"] is False
            assert result["reason"] == "insufficient_samples"

    def test_train_with_samples(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")
            model_dir = os.path.join(tmpdir, "models")
            registry_path = os.path.join(tmpdir, "registry.json")

            # Monkeypatch default paths used by task
            monkeypatch.setattr("src.tasks.ml_training_task.EvaluationDataset", lambda: EvaluationDataset(dataset_dir=dataset_dir))
            monkeypatch.setattr("src.tasks.ml_training_task.ModelTrainer", lambda: __import__("src.ml.model", fromlist=["ModelTrainer"]).ModelTrainer(model_dir=model_dir))
            monkeypatch.setattr("src.tasks.ml_training_task.ModelRegistry", lambda: __import__("src.ml.registry", fromlist=["ModelRegistry"]).ModelRegistry(registry_path=registry_path))

            dataset = EvaluationDataset(dataset_dir=dataset_dir)
            for i in range(15):
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
                        "test_functions": 1 if i > 5 else 0,
                        "docstring_count": i,
                        "type_hint_count": i,
                        "comment_lines": i,
                        "syntax_errors": [],
                        "has_readme": i % 2,
                        "has_requirements": i % 2,
                        "has_tests": 1 if i > 5 else 0,
                    },
                    "test_results": [{"passed": True}] if i > 5 else [],
                    "project": {},
                }
                dataset.append(state, {"overall_score": 50 + i * 3, "passed": i > 5})

            result = train_ml_evaluator.run(min_samples=10, stage="shadow")
            assert result["trained"] is True
            assert "experiment_id" in result
            assert "metrics" in result
