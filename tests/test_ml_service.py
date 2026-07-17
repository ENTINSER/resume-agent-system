"""ML 评估服务单元测试"""

import os
import tempfile

import pytest

from src.ml.service import MLEvaluationService
from src.ml.model import EvalModel, ModelTrainer
from src.ml.dataset import EvaluationDataset
from src.ml.registry import ModelRegistry, STAGE_PRODUCTION
from src.ml.features import FeatureExtractor


class TestMLEvaluationService:
    @pytest.fixture
    def trained_model(self):
        """构造并注册一个 production 模型"""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, "dataset")
            model_dir = os.path.join(tmpdir, "models")
            registry_path = os.path.join(tmpdir, "registry.json")

            dataset = EvaluationDataset(dataset_dir=dataset_dir, version="v1")
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

            trainer = ModelTrainer(model_dir=model_dir)
            report = trainer.train(dataset)

            registry = ModelRegistry(registry_path=registry_path)
            registry.register(
                experiment_id=report["experiment_id"],
                model_path=report["model_path"],
                metrics=report["metrics"],
                stage=STAGE_PRODUCTION,
            )

            yield tmpdir, report["experiment_id"], registry_path

    def test_shadow_mode(self, trained_model, monkeypatch):
        tmpdir, model_id, registry_path = trained_model
        monkeypatch.setenv("ML_EVAL_MODE", "shadow")
        monkeypatch.setenv("ML_EVAL_ENABLED", "true")

        service = MLEvaluationService(mode="shadow", registry_path=registry_path)
        assert service.is_available() is True

        state = {
            "static_metrics": {
                "total_files": 3,
                "python_files": 3,
                "code_lines": 40,
                "functions": 3,
                "classes": 0,
                "imports": 3,
                "test_functions": 1,
                "docstring_count": 2,
                "type_hint_count": 2,
                "comment_lines": 2,
                "syntax_errors": [],
                "has_readme": True,
                "has_requirements": True,
                "has_tests": True,
            },
            "test_results": [{"passed": True}],
            "project": {},
        }
        llm_result = {"overall_score": 80, "passed": True, "source": "llm"}
        result = service.evaluate(state, llm_result)

        assert "ml_score" in result
        assert result["source"] == "llm"
        assert result["overall_score"] == 80

    def test_online_mode_blend(self, trained_model, monkeypatch):
        tmpdir, model_id, registry_path = trained_model
        monkeypatch.setenv("ML_EVAL_MODE", "online")
        monkeypatch.setenv("ML_EVAL_ENABLED", "true")

        service = MLEvaluationService(mode="online", weight_ml=0.3, registry_path=registry_path)
        state = {
            "static_metrics": {
                "total_files": 3,
                "python_files": 3,
                "code_lines": 40,
                "functions": 3,
                "classes": 0,
                "imports": 3,
                "test_functions": 1,
                "docstring_count": 2,
                "type_hint_count": 2,
                "comment_lines": 2,
                "syntax_errors": [],
                "has_readme": True,
                "has_requirements": True,
                "has_tests": True,
            },
            "test_results": [{"passed": True}],
            "project": {},
        }
        llm_result = {"overall_score": 80, "passed": False, "source": "llm"}
        result = service.evaluate(state, llm_result)

        # 当存在严重分歧时，结果会被覆盖为 LLM；否则保持混合
        if result.get("ml_disagreement_level") == "severe":
            assert result["source"] == "llm"
        else:
            assert result["source"] == "hybrid"
            assert result["overall_score"] != 80
        assert "original_llm_score" in result
        assert "ml_disagreement_level" in result
        assert "ml_disagreement_score_diff" in result
        assert "ml_disagreement_action" in result
        assert "ml_disagreement_reason" in result
