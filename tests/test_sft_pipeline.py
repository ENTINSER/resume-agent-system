"""SFT 数据飞轮流水线单元测试"""

import os
import tempfile

import pytest

from src.sft.pipeline import SFTFlywheel
from src.sft.dataset import SFTDataset
from src.sft.collectors import SFTSample


def _make_sample(role: str = "C", instruction: str = "", output: str = "", score: float = 80.0):
    return SFTSample(
        role=role,
        instruction=instruction,
        input_text="",
        output=output,
        context={},
        quality_score=score,
        source_task="t1",
    )


class TestSFTFlywheel:
    def test_process_disabled(self):
        flywheel = SFTFlywheel(enabled=False)
        result = flywheel.process({})
        assert result["enabled"] is False

    def test_process_high_quality_task(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("SFT_ENABLED", "true")
            monkeypatch.setenv("SFT_ROLES", "D,B,E,C")
            monkeypatch.setenv("SFT_MIN_SCORE", "70")

            flywheel = SFTFlywheel(
                enabled=True,
                roles=["D", "B", "E", "C"],
                min_score=70,
            )
            # monkeypatch dataset dir
            flywheel_original = flywheel

            state = {
                "session_id": "s1",
                "task_id": "t1",
                "requirements": "开发一个 Python 计算器",
                "project": {"name": "Calculator", "tech_stack": ["pytest"], "milestones": ["setup", "core"]},
                "code_artifacts": [
                    {"file_path": "main.py", "content": "def add(a, b): return a + b", "language": "python"},
                ],
                "test_results": [{"file": "tests/test_main.py", "passed": True}],
                "static_metrics": {"total_files": 2, "code_lines": 10, "syntax_errors": []},
                "evaluation_result": {"overall_score": 85, "passed": True},
            }

            # Patch dataset path per role
            original_process = flywheel.process

            def patched_process(state):
                # Use temp datasets
                from src.sft.collectors import SFTCollectorPipeline
                from src.sft.quality import QualityGate, Deduplicator
                samples = SFTCollectorPipeline(roles=flywheel.roles).collect(state)
                filtered = QualityGate(min_score=flywheel.min_score).filter_samples(state, samples)
                filtered = Deduplicator.deduplicate(filtered)
                result = {"enabled": True, "roles": {}, "samples_total": 0}
                for role in flywheel.roles:
                    role_samples = [s for s in filtered if s.role == role]
                    if not role_samples:
                        continue
                    dataset = SFTDataset(role=role, dataset_dir=tmpdir, version="v1")
                    added = dataset.add_samples(role_samples)
                    result["roles"][role] = {"added": added, "total": dataset.count()}
                    result["samples_total"] += added
                return result

            result = patched_process(state)
            assert result["samples_total"] > 0
            assert "D" in result["roles"]

    def test_maybe_train_not_enough_samples(self):
        flywheel = SFTFlywheel(enabled=True, roles=["B"])
        flywheel.min_samples = 1000
        result = flywheel.maybe_train("B")
        assert result["trained"] is False

    def test_process_active_learning_reduces_samples(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            flywheel = SFTFlywheel(enabled=True, roles=["C"], min_score=0)
            flywheel.active_learning_enabled = True
            flywheel.active_learning_subset_size = 5

            def make_dataset(role):
                return SFTDataset(role=role, dataset_dir=tmpdir, version="v1")

            monkeypatch.setattr("src.sft.pipeline.SFTDataset", make_dataset)

            state = {
                "session_id": "s1",
                "task_id": "t1",
                "requirements": "开发计算器",
                "project": {"name": "Calc"},
                "code_artifacts": [],
                "test_results": [],
                "static_metrics": {"syntax_errors": []},
                "evaluation_result": {"overall_score": 80, "passed": True},
            }
            # 构造大量 C 角色样本
            samples = [
                _make_sample(role="C", instruction=f"func{i}", output=f"def f{i}(): pass", score=float(i))
                for i in range(20)
            ]
            # 绕过采集器与质量门，直接注入样本
            flywheel.collectors = type(
                "MockPipeline", (), {"collect": lambda self, s: samples}
            )()
            flywheel.quality_gate = type(
                "MockGate", (), {"filter_samples": lambda self, st, sm: sm}
            )()

            result = flywheel.process(state)
            sampled = result["roles"]["C"]["sampled"]
            added = result["roles"]["C"]["added"]
            assert 0 < sampled <= 5
            assert added == sampled
            assert result["samples_total"] == added

    def test_maybe_train_blocked_by_thresholds(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            flywheel = SFTFlywheel(enabled=True, roles=["C"], min_score=0)
            flywheel.per_role_min_samples = 10
            flywheel.min_avg_score = 90.0
            flywheel.max_duplicate_ratio = 0.1

            def make_dataset(role):
                return SFTDataset(role=role, dataset_dir=tmpdir, version="v1")

            monkeypatch.setattr("src.sft.pipeline.SFTDataset", make_dataset)

            # 只添加 3 条低分样本，触发阈值拦截
            dataset = make_dataset("C")
            dataset.add_samples([
                _make_sample(role="C", instruction="a", output="b", score=50.0),
                _make_sample(role="C", instruction="c", output="d", score=55.0),
                _make_sample(role="C", instruction="e", output="f", score=60.0),
            ])

            result = flywheel.maybe_train("C")
            assert result["trained"] is False

    def test_maybe_train_passes_thresholds(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            flywheel = SFTFlywheel(enabled=True, roles=["C"], min_score=0)
            flywheel.per_role_min_samples = 2
            flywheel.min_avg_score = 70.0
            flywheel.max_duplicate_ratio = 0.5

            def make_dataset(role):
                return SFTDataset(role=role, dataset_dir=tmpdir, version="v1")

            monkeypatch.setattr("src.sft.pipeline.SFTDataset", make_dataset)

            dataset = make_dataset("C")
            dataset.add_samples([
                _make_sample(role="C", instruction=f"a{i}", output=f"b{i}", score=80.0)
                for i in range(5)
            ])

            def fake_train(self, role, dataset, output_base_dir="models/sft"):
                return {"output_dir": "/tmp/fake-model"}

            monkeypatch.setattr(
                "src.sft.pipeline.SFTTrainer",
                lambda: type("MockTrainer", (), {"train": fake_train})(),
            )

            result = flywheel.maybe_train("C")
            assert result["trained"] is True
            assert result["role"] == "C"
