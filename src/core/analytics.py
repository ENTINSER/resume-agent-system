"""Analytics utilities for metrics correlation and threshold suggestions.

P2.3 metrics correlation / quality gates / thresholds.
"""

import json
import math
from typing import Any, Dict, List, Optional

try:
    from scipy import stats

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - defensive import
    _HAS_SCIPY = False

try:
    import numpy as np

    _HAS_NUMPY = True
except Exception:  # pragma: no cover - defensive import
    _HAS_NUMPY = False

from src.core.logger import logger


# Metrics that may be present in session state. Extracted opportunistically.
_NUMERIC_METRICS = [
    "overall_score",
    "passed",
    "code_lines",
    "functions",
    "docstring_ratio",
    "type_hint_ratio",
    "graph_max_cyclomatic_complexity",
    "graph_import_cycle_count",
    "graph_test_to_code_function_ratio",
    "test_pass_rate",
    "syntax_error_count",
    "total_files",
    "python_files",
    "classes",
    "imports",
    "test_functions",
    "comment_lines",
]


def _safe_float(value: Any) -> Optional[float]:
    """Convert a value to float if possible, returning None otherwise."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_static_metric(static_metrics: Dict[str, Any], key: str) -> Optional[float]:
    if not isinstance(static_metrics, dict):
        return None
    if key == "syntax_error_count":
        errors = static_metrics.get("syntax_errors", [])
        return float(len(errors)) if isinstance(errors, list) else _safe_float(errors)
    return _safe_float(static_metrics.get(key))


def _extract_session_metrics(session_row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Pull numeric metrics from a sessions table row."""
    state_json = session_row.get("state_json") or "{}"
    try:
        state = json.loads(state_json)
    except Exception:
        state = {}

    eval_result = state.get("evaluation_result") or {}
    if not isinstance(eval_result, dict):
        eval_result = {}

    static_metrics = state.get("static_metrics") or {}
    if not isinstance(static_metrics, dict):
        static_metrics = {}

    test_results = state.get("test_results") or []
    if not isinstance(test_results, list):
        test_results = []

    test_pass_rate = None
    if test_results:
        passed_count = sum(1 for t in test_results if isinstance(t, dict) and t.get("passed", False))
        test_pass_rate = passed_count / len(test_results)

    metrics: Dict[str, Optional[float]] = {
        "overall_score": _safe_float(eval_result.get("overall_score")),
        "passed": 1.0 if eval_result.get("passed", False) else 0.0,
        "test_pass_rate": test_pass_rate,
    }

    for key in _NUMERIC_METRICS:
        if key in ("overall_score", "passed", "test_pass_rate"):
            continue
        metrics[key] = _extract_static_metric(static_metrics, key)

    # Fallbacks from top-level session columns if state JSON is missing values.
    if metrics["overall_score"] is None:
        metrics["overall_score"] = _safe_float(session_row.get("overall_score"))
    if metrics["passed"] is None:
        metrics["passed"] = 1.0 if session_row.get("passed") else 0.0

    return metrics


def _build_matrix(records: List[Dict[str, Optional[float]]]) -> Dict[str, List[Optional[float]]]:
    """Build a column-oriented matrix from metric records, preserving None values.

    Each column is a list of values (float or None). Pairwise complete-case analysis
    is performed later so that sparsely populated metrics do not erase whole rows.
    """
    if not records:
        return {}

    keys = list(records[0].keys())
    matrix: Dict[str, List[Optional[float]]] = {k: [] for k in keys}

    for record in records:
        for k in keys:
            value = record.get(k)
            if isinstance(value, float) and math.isnan(value):
                value = None
            matrix[k].append(value)

    return matrix


def _numpy_pearson(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation using numpy."""
    if not _HAS_NUMPY:
        return float("nan")
    a = np.array(x, dtype=float)
    b = np.array(y, dtype=float)
    if a.size < 2 or b.size < 2 or a.std(ddof=0) == 0 or b.std(ddof=0) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _numpy_spearman(x: List[float], y: List[float]) -> float:
    """Compute Spearman correlation using numpy (rank-based fallback)."""
    if not _HAS_NUMPY:
        return float("nan")

    def ranks(values: List[float]) -> List[float]:
        sorted_vals = sorted((v, i) for i, v in enumerate(values))
        rank = [0.0] * len(values)
        i = 0
        while i < len(sorted_vals):
            j = i
            while j + 1 < len(sorted_vals) and sorted_vals[j + 1][0] == sorted_vals[i][0]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                rank[sorted_vals[k][1]] = avg_rank
            i = j + 1
        return rank

    if len(x) < 2 or len(y) < 2:
        return 0.0
    return _numpy_pearson(ranks(x), ranks(y))


def _pairwise_complete(x: List[Optional[float]], y: List[Optional[float]]) -> List[tuple]:
    """Return pairs of (x, y) where both values are present."""
    return [(a, b) for a, b in zip(x, y) if a is not None and b is not None]


def _is_constant(values) -> bool:
    """Return True if all values are identical."""
    if len(values) < 2:
        return True
    first = values[0]
    return all(v == first for v in values)


def _correlation_matrix(matrix: Dict[str, List[Optional[float]]]) -> Dict[str, Dict[str, float]]:
    """Compute Pearson and Spearman correlation matrices using pairwise complete cases."""
    keys = list(matrix.keys())
    n_total = len(matrix.get(keys[0], [])) if keys else 0

    pearson: Dict[str, Dict[str, float]] = {k: {} for k in keys}
    spearman: Dict[str, Dict[str, float]] = {k: {} for k in keys}

    if n_total < 2:
        return {"pearson": pearson, "spearman": spearman}

    for i, ki in enumerate(keys):
        for kj in keys[i:]:
            pairs = _pairwise_complete(matrix[ki], matrix[kj])
            if len(pairs) < 2:
                p_val = 0.0
                s_val = 0.0
            else:
                x, y = zip(*pairs)
                # Constant inputs make correlation undefined; return 0.0 to avoid scipy warnings.
                if _is_constant(x) or _is_constant(y):
                    p_val = 0.0
                    s_val = 0.0
                elif _HAS_SCIPY:
                    try:
                        p_val = float(stats.pearsonr(x, y)[0])
                    except Exception:
                        p_val = _numpy_pearson(x, y)
                    try:
                        s_val = float(stats.spearmanr(x, y)[0])
                    except Exception:
                        s_val = _numpy_spearman(x, y)
                else:
                    p_val = _numpy_pearson(x, y)
                    s_val = _numpy_spearman(x, y)

            pearson[ki][kj] = p_val
            pearson[kj][ki] = p_val
            spearman[ki][kj] = s_val
            spearman[kj][ki] = s_val

    return {"pearson": pearson, "spearman": spearman}


def compute_metric_correlations(store: Any = None, min_samples: int = 10) -> Dict[str, Any]:
    """Compute correlation matrices across historical sessions.

    Args:
        store: Optional SessionStore instance. If None, a default store is created.
        min_samples: Minimum number of sessions required before computing correlations.

    Returns:
        A dictionary with correlation matrices, sample count, and top predictors.
    """
    result: Dict[str, Any] = {
        "correlations": {},
        "sample_count": 0,
        "top_predictors": [],
        "error": None,
    }

    try:
        if store is None:
            from src.core.persistence import get_session_store

            store = get_session_store()

        sessions = store.list_sessions(limit=1000)
    except Exception as e:
        logger.warning(f"[Analytics] Failed to list sessions: {e}")
        result["error"] = str(e)
        return result

    records: List[Dict[str, Optional[float]]] = []
    for session in sessions:
        try:
            metrics = _extract_session_metrics(session)
        except Exception:
            continue
        records.append(metrics)

    matrix = _build_matrix(records)
    # Sample count reflects sessions that have a pass/fail label.
    passed_values = matrix.get("passed", []) if matrix else []
    sample_count = sum(1 for v in passed_values if v is not None)
    result["sample_count"] = sample_count

    if sample_count < min_samples:
        logger.info(
            f"[Analytics] Insufficient samples ({sample_count} < {min_samples}), skipping correlation"
        )
        return result

    correlations = _correlation_matrix(matrix)
    result["correlations"] = correlations

    # Identify metrics most correlated with "passed".
    if "passed" in correlations["pearson"]:
        passed_corr = correlations["pearson"]["passed"]
        sorted_predictors = sorted(
            ((metric, abs(corr)) for metric, corr in passed_corr.items() if metric != "passed"),
            key=lambda item: item[1],
            reverse=True,
        )
        result["top_predictors"] = [
            {"metric": metric, "abs_correlation": corr, "correlation": passed_corr[metric]}
            for metric, corr in sorted_predictors
        ]

    return result


def export_threshold_suggestions(correlations: Dict[str, Any]) -> Dict[str, Any]:
    """Derive simple threshold recommendations from correlation output.

    The suggestions are intentionally conservative and meant to seed gate tuning.
    """
    suggestions: Dict[str, Any] = {
        "score_cutoff": None,
        "syntax_error_allowance": None,
        "test_pass_rate_cutoff": None,
        "notes": [],
    }

    pearson = correlations.get("correlations", {}).get("pearson", {})
    top_predictors = correlations.get("top_predictors", [])

    # Default score cutoff when no data is available.
    if not pearson:
        suggestions["notes"].append("No correlation data available; using defaults.")
        suggestions["score_cutoff"] = 80.0
        suggestions["syntax_error_allowance"] = 0
        suggestions["test_pass_rate_cutoff"] = 1.0
        return suggestions

    # Score cutoff suggestion: preserve default unless evidence suggests otherwise.
    suggestions["score_cutoff"] = 80.0
    suggestions["syntax_error_allowance"] = 0
    suggestions["test_pass_rate_cutoff"] = 1.0

    passed_corr = pearson.get("passed", {})
    if "overall_score" in passed_corr:
        corr = passed_corr["overall_score"]
        suggestions["notes"].append(
            f"overall_score correlation with passed: {corr:.3f}"
        )
        if corr < 0:
            suggestions["notes"].append(
                "overall_score is negatively correlated with pass status; review data quality."
            )

    if top_predictors:
        suggestions["notes"].append(
            f"Top predictor: {top_predictors[0]['metric']} "
            f"(r={top_predictors[0]['correlation']:.3f})"
        )

    if "syntax_error_count" in passed_corr and passed_corr["syntax_error_count"] < 0:
        suggestions["notes"].append(
            "syntax_error_count negatively correlates with passes; keep allowance at 0."
        )

    if "test_pass_rate" in passed_corr and passed_corr["test_pass_rate"] > 0:
        suggestions["notes"].append(
            "test_pass_rate positively correlates with passes; maintain strict cutoff."
        )

    return suggestions
