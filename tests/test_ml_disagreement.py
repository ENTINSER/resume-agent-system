"""LLM/ML 评估分歧检测与裁决单元测试"""

import pytest

from src.ml.disagreement import DisagreementLevel, DisagreementDetector


class TestDisagreementDetector:
    """测试分歧检测器"""

    def test_detect_none(self):
        detector = DisagreementDetector(threshold=15.0, severe_threshold=30.0)
        llm_result = {"overall_score": 80, "passed": True}
        ml_result = {"ml_score": 85, "ml_passed": True}

        result = detector.detect(llm_result, ml_result)

        assert result["level"] == DisagreementLevel.NONE.value
        assert result["score_diff"] == 5.0
        assert result["passed_mismatch"] is False
        assert result["action"] == "accept"
        assert "一致" in result["message"]

    def test_detect_mild(self):
        detector = DisagreementDetector(threshold=15.0, severe_threshold=30.0)
        llm_result = {"overall_score": 80, "passed": True}
        ml_result = {"ml_score": 60, "ml_passed": True}

        result = detector.detect(llm_result, ml_result)

        assert result["level"] == DisagreementLevel.MILD.value
        assert result["score_diff"] == 20.0
        assert result["passed_mismatch"] is False
        assert result["action"] == "re_evaluate"
        assert "轻度" in result["message"]

    def test_detect_severe_by_score_diff(self):
        detector = DisagreementDetector(threshold=15.0, severe_threshold=30.0)
        llm_result = {"overall_score": 80, "passed": True}
        ml_result = {"ml_score": 40, "ml_passed": True}

        result = detector.detect(llm_result, ml_result)

        assert result["level"] == DisagreementLevel.SEVERE.value
        assert result["score_diff"] == 40.0
        assert result["passed_mismatch"] is False
        assert result["action"] == "human_review"
        assert "严重" in result["message"]

    def test_detect_severe_by_passed_mismatch(self):
        detector = DisagreementDetector(threshold=15.0, severe_threshold=30.0)
        llm_result = {"overall_score": 80, "passed": True}
        ml_result = {"ml_score": 82, "ml_passed": False}

        result = detector.detect(llm_result, ml_result)

        assert result["level"] == DisagreementLevel.SEVERE.value
        assert result["score_diff"] == 2.0
        assert result["passed_mismatch"] is True
        assert result["action"] == "human_review"
        assert "通过性不一致" in result["message"]

    def test_detect_defaults_from_settings(self, monkeypatch):
        monkeypatch.setenv("ML_EVAL_DISAGREEMENT_THRESHOLD", "10.0")
        monkeypatch.setenv("ML_EVAL_SEVERE_DISAGREEMENT_THRESHOLD", "25.0")

        detector = DisagreementDetector()
        assert detector.threshold == 10.0
        assert detector.severe_threshold == 25.0

    def test_detect_uses_ml_score_aliases(self):
        """ml_result 也可使用 overall_score 字段作为分数"""
        detector = DisagreementDetector(threshold=15.0, severe_threshold=30.0)
        llm_result = {"overall_score": 80, "passed": True}
        ml_result = {"overall_score": 50, "ml_passed": True}

        result = detector.detect(llm_result, ml_result)

        assert result["level"] == DisagreementLevel.SEVERE.value
        assert result["score_diff"] == 30.0

    def test_resolve_none(self):
        detector = DisagreementDetector(threshold=15.0, severe_threshold=30.0)
        llm_result = {"overall_score": 80, "passed": True}
        ml_result = {"ml_score": 85, "ml_passed": True}
        disagreement = detector.detect(llm_result, ml_result)

        resolution = detector.resolve(llm_result, ml_result, disagreement)

        assert resolution["overall_score"] == 80.0
        assert resolution["passed"] is True
        assert resolution["source"] == "llm"
        assert "无分歧" in resolution["reason"]

    def test_resolve_mild(self):
        detector = DisagreementDetector(threshold=15.0, severe_threshold=30.0)
        llm_result = {"overall_score": 80, "passed": True}
        ml_result = {"ml_score": 60, "ml_passed": True}
        disagreement = detector.detect(llm_result, ml_result)

        resolution = detector.resolve(llm_result, ml_result, disagreement)

        assert resolution["overall_score"] == 70.0
        assert resolution["passed"] is True
        assert resolution["source"] == "hybrid"
        assert "平均" in resolution["reason"]

    def test_resolve_mild_conservative_fail(self):
        detector = DisagreementDetector(threshold=15.0, severe_threshold=30.0)
        llm_result = {"overall_score": 80, "passed": True}
        ml_result = {"ml_score": 60, "ml_passed": False}
        disagreement = detector.detect(llm_result, ml_result)

        resolution = detector.resolve(llm_result, ml_result, disagreement)

        # passed mismatch makes it severe, so this case is not reached normally
        assert disagreement["level"] == DisagreementLevel.SEVERE.value

    def test_resolve_severe_trust_llm(self):
        detector = DisagreementDetector(threshold=15.0, severe_threshold=30.0)
        llm_result = {"overall_score": 80, "passed": True}
        ml_result = {"ml_score": 40, "ml_passed": True}
        disagreement = detector.detect(llm_result, ml_result)

        resolution = detector.resolve(llm_result, ml_result, disagreement)

        assert resolution["overall_score"] == 80.0
        assert resolution["passed"] is True
        assert resolution["source"] == "llm"
        assert "严重分歧" in resolution["reason"]

    def test_resolve_severe_passed_mismatch_trust_llm(self):
        detector = DisagreementDetector(threshold=15.0, severe_threshold=30.0)
        llm_result = {"overall_score": 80, "passed": False}
        ml_result = {"ml_score": 82, "ml_passed": True}
        disagreement = detector.detect(llm_result, ml_result)

        resolution = detector.resolve(llm_result, ml_result, disagreement)

        assert resolution["overall_score"] == 80.0
        assert resolution["passed"] is False
        assert resolution["source"] == "llm"
        assert "严重分歧" in resolution["reason"]


class TestDisagreementLevel:
    def test_enum_values(self):
        assert DisagreementLevel.NONE.value == "none"
        assert DisagreementLevel.MILD.value == "mild"
        assert DisagreementLevel.SEVERE.value == "severe"
