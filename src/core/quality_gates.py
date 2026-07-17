"""Quality gates and multi-metric thresholds.

P2.3 metrics correlation / quality gates / thresholds.
"""

from typing import Any, Dict, List, Optional

from src.core.config import settings
from src.core.logger import logger


class MultiMetricGate:
    """Composite quality gate that checks evaluation, static, test, and ML disagreement metrics.

    Thresholds default to platform settings but can be overridden per instance.
    """

    # Ordering used to compare disagreement severity.
    DISAGREEMENT_LEVELS = ["none", "mild", "severe"]

    def __init__(
        self,
        min_score: Optional[float] = None,
        require_passed: bool = True,
        max_syntax_errors: int = 0,
        min_test_pass_rate: float = 1.0,
        max_disagreement_level: str = "mild",
        max_static_risk_score: Optional[float] = None,
    ):
        self.min_score = min_score if min_score is not None else settings.eval_pass_threshold
        self.require_passed = require_passed
        self.max_syntax_errors = max_syntax_errors
        self.min_test_pass_rate = min_test_pass_rate
        self.max_disagreement_level = max_disagreement_level
        # A very high default effectively disables the risk-score gate unless configured.
        self.max_static_risk_score = max_static_risk_score if max_static_risk_score is not None else 1e9

    def _disagreement_index(self, level: str) -> int:
        try:
            return self.DISAGREEMENT_LEVELS.index(level)
        except ValueError:
            # Unknown levels are treated as most severe.
            return len(self.DISAGREEMENT_LEVELS)

    def _compute_test_pass_rate(self, test_results: List[Dict[str, Any]]) -> float:
        if not test_results:
            # No tests ran; default to passing only when zero tests are required.
            return 1.0
        passed = sum(1 for t in test_results if t.get("passed", False))
        return passed / len(test_results)

    def _compute_static_risk_score(self, static_metrics: Dict[str, Any]) -> float:
        cc = static_metrics.get("graph_max_cyclomatic_complexity", 0) or 0
        cycles = static_metrics.get("graph_import_cycle_count", 0) or 0
        syntax_errors = static_metrics.get("syntax_errors", []) or []
        return cc / 10.0 + cycles * 5.0 + len(syntax_errors) * 10.0

    def check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all configured gates against the supplied state.

        Returns a structured report containing the decision, blocking reasons,
        and the intermediate metric values used for the decision.
        """
        eval_result = state.get("evaluation_result") or {}
        static_metrics = state.get("static_metrics") or {}
        test_results = state.get("test_results") or []

        score = float(eval_result.get("overall_score", 0.0) if isinstance(eval_result, dict) else 0.0)
        passed_flag = bool(eval_result.get("passed", False)) if isinstance(eval_result, dict) else False

        syntax_errors = static_metrics.get("syntax_errors", []) or []
        syntax_error_count = len(syntax_errors)

        test_pass_rate = self._compute_test_pass_rate(test_results)
        static_risk_score = self._compute_static_risk_score(static_metrics)

        disagreement_level = state.get("ml_disagreement_level", "none")
        if not isinstance(disagreement_level, str) or not disagreement_level:
            disagreement_level = "none"

        reasons: List[str] = []

        # Score gate
        if score < self.min_score:
            reasons.append(
                f"Overall score {score:.2f} below required minimum {self.min_score:.2f}"
            )

        # Passed gate
        if self.require_passed and not passed_flag:
            reasons.append("Evaluation did not pass")

        # Syntax errors gate
        if syntax_error_count > self.max_syntax_errors:
            reasons.append(
                f"Too many syntax errors: {syntax_error_count} > {self.max_syntax_errors}"
            )

        # Test pass-rate gate
        if test_pass_rate < self.min_test_pass_rate:
            reasons.append(
                f"Test pass rate {test_pass_rate:.2f} below required {self.min_test_pass_rate:.2f}"
            )

        # Static risk gate
        if static_risk_score > self.max_static_risk_score:
            reasons.append(
                f"Static risk score {static_risk_score:.2f} exceeds maximum {self.max_static_risk_score:.2f}"
            )

        # ML disagreement gate. Severe disagreement always rejects unless explicitly allowed.
        if self._disagreement_index(disagreement_level) > self._disagreement_index(self.max_disagreement_level):
            reasons.append(
                f"ML disagreement level '{disagreement_level}' exceeds allowed '{self.max_disagreement_level}'"
            )

        approved = not reasons

        if not approved:
            logger.info(
                f"[MultiMetricGate] Rejected: {reasons}",
                extra={
                    "score": score,
                    "test_pass_rate": test_pass_rate,
                    "static_risk_score": static_risk_score,
                    "disagreement_level": disagreement_level,
                },
            )

        return {
            "approved": approved,
            "reasons": reasons,
            "score": score,
            "test_pass_rate": test_pass_rate,
            "static_risk_score": static_risk_score,
            "disagreement_level": disagreement_level,
        }
