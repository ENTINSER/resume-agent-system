"""Tests for MultiMetricGate."""

import pytest

from src.core.quality_gates import MultiMetricGate


class TestMultiMetricGate:
    def test_approved_default_state(self):
        gate = MultiMetricGate()
        state = {
            "evaluation_result": {"overall_score": 85.0, "passed": True},
            "static_metrics": {
                "syntax_errors": [],
                "graph_max_cyclomatic_complexity": 5,
                "graph_import_cycle_count": 0,
            },
            "test_results": [{"passed": True}],
        }
        result = gate.check(state)
        assert result["approved"] is True
        assert result["reasons"] == []
        assert result["score"] == 85.0
        assert result["test_pass_rate"] == 1.0
        assert result["static_risk_score"] == 0.5
        assert result["disagreement_level"] == "none"

    def test_rejected_low_score(self):
        gate = MultiMetricGate(min_score=80.0)
        state = {
            "evaluation_result": {"overall_score": 75.0, "passed": True},
            "static_metrics": {"syntax_errors": []},
            "test_results": [{"passed": True}],
        }
        result = gate.check(state)
        assert result["approved"] is False
        assert any("score" in reason.lower() for reason in result["reasons"])

    def test_rejected_not_passed(self):
        gate = MultiMetricGate()
        state = {
            "evaluation_result": {"overall_score": 85.0, "passed": False},
            "static_metrics": {"syntax_errors": []},
            "test_results": [{"passed": True}],
        }
        result = gate.check(state)
        assert result["approved"] is False
        assert any("did not pass" in reason.lower() for reason in result["reasons"])

    def test_rejected_syntax_errors(self):
        gate = MultiMetricGate(max_syntax_errors=0)
        state = {
            "evaluation_result": {"overall_score": 85.0, "passed": True},
            "static_metrics": {"syntax_errors": ["error1"]},
            "test_results": [{"passed": True}],
        }
        result = gate.check(state)
        assert result["approved"] is False
        assert any("syntax" in reason.lower() for reason in result["reasons"])
        assert result["static_risk_score"] == 10.0

    def test_rejected_test_pass_rate(self):
        gate = MultiMetricGate(min_test_pass_rate=1.0)
        state = {
            "evaluation_result": {"overall_score": 85.0, "passed": True},
            "static_metrics": {"syntax_errors": []},
            "test_results": [
                {"passed": True},
                {"passed": False},
            ],
        }
        result = gate.check(state)
        assert result["approved"] is False
        assert result["test_pass_rate"] == 0.5
        assert any("test pass rate" in reason.lower() for reason in result["reasons"])

    def test_severe_disagreement_rejects(self):
        gate = MultiMetricGate(max_disagreement_level="mild")
        state = {
            "evaluation_result": {"overall_score": 85.0, "passed": True},
            "static_metrics": {"syntax_errors": []},
            "test_results": [{"passed": True}],
            "ml_disagreement_level": "severe",
        }
        result = gate.check(state)
        assert result["approved"] is False
        assert any("severe" in reason.lower() for reason in result["reasons"])

    def test_severe_allowed_when_configured(self):
        gate = MultiMetricGate(max_disagreement_level="severe")
        state = {
            "evaluation_result": {"overall_score": 85.0, "passed": True},
            "static_metrics": {"syntax_errors": []},
            "test_results": [{"passed": True}],
            "ml_disagreement_level": "severe",
        }
        result = gate.check(state)
        assert result["approved"] is True

    def test_mild_disagreement_allowed(self):
        gate = MultiMetricGate(max_disagreement_level="mild")
        state = {
            "evaluation_result": {"overall_score": 85.0, "passed": True},
            "static_metrics": {"syntax_errors": []},
            "test_results": [{"passed": True}],
            "ml_disagreement_level": "mild",
        }
        result = gate.check(state)
        assert result["approved"] is True

    def test_static_risk_score_gate(self):
        gate = MultiMetricGate(max_static_risk_score=1.0)
        state = {
            "evaluation_result": {"overall_score": 85.0, "passed": True},
            "static_metrics": {
                "syntax_errors": [],
                "graph_max_cyclomatic_complexity": 20,
                "graph_import_cycle_count": 0,
            },
            "test_results": [{"passed": True}],
        }
        result = gate.check(state)
        assert result["approved"] is False
        assert result["static_risk_score"] == 2.0
        assert any("risk" in reason.lower() for reason in result["reasons"])

    def test_missing_evaluation_result(self):
        gate = MultiMetricGate()
        state = {
            "static_metrics": {"syntax_errors": []},
            "test_results": [{"passed": True}],
        }
        result = gate.check(state)
        assert result["approved"] is False
        assert result["score"] == 0.0

    def test_no_tests_required(self):
        gate = MultiMetricGate(min_test_pass_rate=1.0)
        state = {
            "evaluation_result": {"overall_score": 85.0, "passed": True},
            "static_metrics": {"syntax_errors": []},
            "test_results": [],
        }
        result = gate.check(state)
        assert result["approved"] is True
        assert result["test_pass_rate"] == 1.0
