"""Tests for analytics correlation and threshold suggestion utilities."""

import json
import os
import tempfile

import pytest

from src.core.analytics import compute_metric_correlations, export_threshold_suggestions
from src.core.persistence import SessionStore


class FakeStore:
    """In-memory stand-in for SessionStore."""

    def __init__(self, sessions):
        self._sessions = sessions

    def list_sessions(self, limit=50):
        return self._sessions[:limit]


class TestAnalytics:
    def test_empty_store_returns_empty_result(self):
        store = FakeStore([])
        result = compute_metric_correlations(store=store, min_samples=5)
        assert result["sample_count"] == 0
        assert result["correlations"] == {}
        assert result["top_predictors"] == []

    def test_insufficient_samples_returns_empty_correlations(self):
        store = FakeStore([
            {"session_id": "s1", "state_json": json.dumps({"evaluation_result": {"overall_score": 80, "passed": True}})}
        ])
        result = compute_metric_correlations(store=store, min_samples=5)
        assert result["sample_count"] < 5
        assert result["correlations"] == {}

    def test_synthetic_sessions_correlation(self):
        sessions = []
        for i in range(12):
            passed = i >= 6
            sessions.append({
                "session_id": f"s{i}",
                "state_json": json.dumps({
                    "evaluation_result": {
                        "overall_score": 50 + i * 4 if not passed else 80 + (i - 6) * 2,
                        "passed": passed,
                    },
                    "static_metrics": {
                        "code_lines": 20 + i * 5,
                        "functions": 2 + i,
                        "docstring_ratio": 0.1 * i,
                        "type_hint_ratio": 0.05 * i,
                        "syntax_errors": [],
                        "graph_max_cyclomatic_complexity": i,
                        "graph_import_cycle_count": 0,
                        "graph_test_to_code_function_ratio": 0.5 if passed else 0.1,
                    },
                    "test_results": [{"passed": True}] if passed else [{"passed": False}],
                })
            })

        store = FakeStore(sessions)
        result = compute_metric_correlations(store=store, min_samples=10)
        assert result["sample_count"] >= 10
        assert "pearson" in result["correlations"]
        assert "spearman" in result["correlations"]
        assert result["top_predictors"]
        passed_corr = result["correlations"]["pearson"]["passed"]
        assert "overall_score" in passed_corr

    def test_export_threshold_suggestions_defaults(self):
        suggestions = export_threshold_suggestions({})
        assert suggestions["score_cutoff"] == 80.0
        assert suggestions["syntax_error_allowance"] == 0
        assert suggestions["test_pass_rate_cutoff"] == 1.0

    def test_export_threshold_suggestions_with_correlations(self):
        correlations = {
            "correlations": {
                "pearson": {
                    "passed": {
                        "overall_score": 0.9,
                        "syntax_error_count": -0.7,
                        "test_pass_rate": 0.85,
                    }
                }
            },
            "top_predictors": [
                {"metric": "overall_score", "abs_correlation": 0.9, "correlation": 0.9}
            ],
        }
        suggestions = export_threshold_suggestions(correlations)
        assert suggestions["score_cutoff"] == 80.0
        assert any("overall_score correlation" in note for note in suggestions["notes"])

    def test_real_session_store_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "sessions.db")
            store = SessionStore(db_path=db_path)

            for i in range(15):
                passed = i % 2 == 0
                state = {
                    "session_id": f"s{i}",
                    "requirements": "test",
                    "status": "complete",
                    "evaluation_result": {
                        "overall_score": 80.0 if passed else 40.0,
                        "passed": passed,
                    },
                    "static_metrics": {
                        "syntax_errors": [],
                        "code_lines": 50 + i,
                    },
                    "test_results": [{"passed": passed}],
                }
                store.save_session(state)

            result = compute_metric_correlations(store=store, min_samples=10)
            assert result["sample_count"] >= 10
            assert "passed" in result["correlations"]["pearson"]
