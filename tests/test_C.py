"""C 智能体（训练工程师）单元测试"""

import os
import json
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from src.core.C import (
    trainer_node,
    _should_train,
    MLXTrainer,
    DatasetBuilder,
)
from src.core.shared_state import create_initial_state
from src.core.config import Settings


@pytest.fixture
def training_state():
    state = create_initial_state("微调一个 LangGraph 多智能体写作系统")
    state["code_artifacts"] = [
        {
            "file_path": "main.py",
            "language": "python",
            "content": (
                'def hello(name: str) -> str:\n'
                '    """问候"""\n'
                '    return f"Hello, {name}"\n\n'
                'def add(a: int, b: int) -> int:\n'
                '    """返回两数之和"""\n'
                '    return a + b\n\n'
                'def mul(a: int, b: int) -> int:\n'
                '    """返回两数之积"""\n'
                '    return a * b\n'
            ),
        }
    ]
    state["evaluation_result"] = {"passed": True, "overall_score": 88}
    return state


class TestShouldTrain:
    def test_keyword_train(self):
        assert _should_train(create_initial_state("训练一个模型")) is True

    def test_keyword_lora(self):
        assert _should_train(create_initial_state("使用 LoRA 微调")) is True

    def test_no_keyword(self):
        assert _should_train(create_initial_state("开发一个计算器")) is False


class TestMLXTrainer:
    def test_generate_training_script(self):
        script = MLXTrainer.generate_training_script(
            model_name="mlx-community/test",
            train_data="data/train.jsonl",
            val_data="data/val.jsonl",
            output_dir="models/test",
        )
        assert "mlx.core" in script
        assert "data/train.jsonl" in script
        assert "models/test" in script

    def test_check_mlx_available(self):
        result = MLXTrainer.check_mlx_available()
        assert isinstance(result, dict)
        assert "mlx" in result
        assert "mlx_lm" in result


class TestDatasetBuilder:
    def test_from_code_artifacts(self):
        artifacts = [
            {
                "file_path": "main.py",
                "language": "python",
                "content": 'def add(a: int, b: int) -> int:\n    """返回两数之和"""\n    return a + b\n',
            }
        ]
        samples = DatasetBuilder.from_code_artifacts(artifacts)
        assert len(samples) >= 1
        assert "instruction" in samples[0]
        assert "output" in samples[0]

    def test_save_and_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "train.jsonl")
            samples = [
                {"instruction": "a", "input": "", "output": "b"},
                {"instruction": "c", "input": "", "output": "d"},
                {"instruction": "e", "input": "", "output": "f"},
                {"instruction": "g", "input": "", "output": "h"},
            ]
            DatasetBuilder.save_to_jsonl(samples, path)
            assert os.path.exists(path)

            train_path, val_path, train_count, val_count = DatasetBuilder.split_dataset(path)
            assert os.path.exists(train_path)
            assert os.path.exists(val_path)
            assert train_count + val_count == 4


class TestTrainerNode:
    def test_skip_when_no_train_keyword(self):
        state = create_initial_state("开发一个 Python 计算器")
        result = trainer_node(state)
        assert result["status"] == "training_skipped"

    def test_insufficient_data(self):
        state = create_initial_state("训练一个模型")
        result = trainer_node(state)
        assert result["status"] == "training_insufficient_data"

    def test_generates_artifacts(self, training_state, monkeypatch, tmp_path):
        monkeypatch.setattr("src.core.C.settings", Settings(sft_auto_train=False))

        with patch("src.core.C.MLXTrainer.check_mlx_available") as mock_mlx:
            mock_mlx.return_value = {"mlx": True, "mlx_lm": True, "gpu": False, "recommendation": ""}
            result = trainer_node(training_state)

        assert result["status"] in ("script_ready", "training")
        jobs = result.get("training_jobs", [])
        assert len(jobs) >= 1
        assert jobs[-1]["status"] in ("script_ready", "training")

    def test_auto_train_starts_background(self, training_state, monkeypatch, tmp_path):
        """sft_auto_train=true 且 MLX 就绪时应启动后台训练"""
        monkeypatch.setattr("src.core.C.settings", Settings(sft_auto_train=True))

        with patch("src.core.C.MLXTrainer.check_mlx_available") as mock_mlx, \
             patch("src.core.C._run_mlx_training") as mock_run:
            mock_mlx.return_value = {"mlx": True, "mlx_lm": True, "gpu": False, "recommendation": ""}
            mock_run.return_value = training_state
            result = trainer_node(training_state)

        mock_run.assert_called_once()
