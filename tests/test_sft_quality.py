"""SFT 数据质量中心单元测试"""

import pytest

from src.sft.quality import QualityGate, Deduplicator
from src.sft.collectors import SFTSample


class TestQualityGate:
    def test_state_filter_passed(self):
        gate = QualityGate(min_score=70, require_passed=True)
        state = {
            "evaluation_result": {"overall_score": 85, "passed": True},
            "test_results": [{"passed": True}],
            "static_metrics": {"syntax_errors": []},
        }
        assert gate.check_state(state)["approved"] is True

    def test_state_filter_low_score(self):
        gate = QualityGate(min_score=70, require_passed=True)
        state = {
            "evaluation_result": {"overall_score": 60, "passed": False},
            "test_results": [],
            "static_metrics": {"syntax_errors": []},
        }
        result = gate.check_state(state)
        assert result["approved"] is False
        assert any("score" in r for r in result["reasons"])

    def test_sample_secret_detection(self):
        gate = QualityGate(enable_secret_scan=True)
        sample = SFTSample(
            role="B",
            instruction="写代码",
            input_text="",
            output="api_key = 'sk-123456789012345678901234567890123456789012345678'",
            context={},
        )
        check = gate.check_sample(sample)
        assert check["safe"] is False
        assert any("OPENAI_API_KEY" in issue for issue in check["issues"])

    def test_sample_pii_detection(self):
        gate = QualityGate(enable_pii_scan=True)
        sample = SFTSample(
            role="B",
            instruction="写代码",
            input_text="",
            output="phone = '13800138000'",
            context={},
        )
        check = gate.check_sample(sample)
        assert check["safe"] is False


class TestDeduplicator:
    def test_deduplicate(self):
        samples = [
            SFTSample(role="C", instruction="a", input_text="", output="b", context={}),
            SFTSample(role="C", instruction="a", input_text="", output="b", context={}),
            SFTSample(role="C", instruction="c", input_text="", output="d", context={}),
        ]
        unique = Deduplicator.deduplicate(samples)
        assert len(unique) == 2


class TestDistributionCheck:
    def test_check_distribution(self):
        samples = [
            SFTSample(role="C", instruction="a", input_text="", output="b", context={}, quality_score=80.0),
            SFTSample(role="C", instruction="a", input_text="", output="b", context={}, quality_score=90.0),
            SFTSample(role="D", instruction="c", input_text="", output="d", context={}, quality_score=70.0),
        ]
        gate = QualityGate()
        dist = gate.check_distribution(samples)
        assert dist["avg_score"] == pytest.approx(80.0)
        assert dist["duplicate_ratio"] == pytest.approx(1 / 3)
        assert dist["role_counts"] == {"C": 2, "D": 1}

    def test_validate_distribution_passes(self):
        gate = QualityGate(per_role_min_samples=2, max_duplicate_ratio=0.5)
        dist = {
            "avg_score": 80.0,
            "duplicate_ratio": 0.2,
            "role_counts": {"C": 3},
        }
        result = gate.validate_distribution(dist, "C")
        assert result["passed"] is True
        assert result["reasons"] == []

    def test_validate_distribution_fails(self):
        gate = QualityGate(per_role_min_samples=5, max_duplicate_ratio=0.1)
        dist = {
            "avg_score": 60.0,
            "duplicate_ratio": 0.5,
            "role_counts": {"C": 2},
        }
        result = gate.validate_distribution(dist, "C")
        assert result["passed"] is False
        assert any("samples_below_threshold" in r for r in result["reasons"])
        assert any("duplicate_ratio_above_threshold" in r for r in result["reasons"])

    def test_check_distribution_empty(self):
        gate = QualityGate()
        dist = gate.check_distribution([])
        assert dist["avg_score"] == 0.0
        assert dist["duplicate_ratio"] == 0.0
        assert dist["role_counts"] == {}
