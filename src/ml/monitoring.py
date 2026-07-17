"""ML 评估监控与漂移检测

记录预测分布、简单统计量，并提供与 Prometheus 对齐的指标输出。
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.logger import logger


DEFAULT_MONITORING_PATH = "data/ml_monitoring.jsonl"


class ModelMonitor:
    """模型监控器"""

    def __init__(self, monitoring_path: str = DEFAULT_MONITORING_PATH):
        self.monitoring_path = Path(monitoring_path)
        self.monitoring_path.parent.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        model_id: str,
        features: Dict[str, Any],
        ml_score: float,
        ml_passed: bool,
    ) -> None:
        """记录一次预测"""
        record = {
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "ml_score": float(ml_score),
            "ml_passed": bool(ml_passed),
            "feature_summary": {
                "code_lines": features.get("code_lines", 0),
                "test_pass_rate": features.get("test_pass_rate", 0),
                "syntax_errors": features.get("syntax_errors", 0),
            },
        }
        with open(self.monitoring_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def get_recent_scores(self, window: int = 100) -> List[float]:
        """获取最近 window 条预测分数"""
        scores = []
        if not self.monitoring_path.exists():
            return scores
        with open(self.monitoring_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    scores.append(record["ml_score"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return scores[-window:]

    def detect_drift(self, baseline_scores: List[float], current_scores: List[float]) -> Dict[str, Any]:
        """基于均值漂移做简单检测

        后续可替换为 PSI 或 KS 检验。
        """
        if not baseline_scores or not current_scores:
            return {"drift_detected": False, "reason": "样本不足"}

        baseline_mean = sum(baseline_scores) / len(baseline_scores)
        current_mean = sum(current_scores) / len(current_scores)
        std = math.sqrt(sum((x - baseline_mean) ** 2 for x in baseline_scores) / max(len(baseline_scores), 1))

        # 如果当前均值偏离基线超过 1.5 个标准差，认为漂移
        threshold = 1.5 * std if std > 0 else 5.0
        drift_detected = abs(current_mean - baseline_mean) > threshold

        return {
            "drift_detected": drift_detected,
            "baseline_mean": round(baseline_mean, 4),
            "current_mean": round(current_mean, 4),
            "std": round(std, 4),
            "threshold": round(threshold, 4),
        }

    def prometheus_metrics(self) -> str:
        """输出 Prometheus 格式的简单指标"""
        scores = self.get_recent_scores(window=1000)
        total = len(scores)
        if not scores:
            return "# 暂无 ML 预测数据\n"

        avg_score = sum(scores) / total
        lines = [
            "# HELP ml_eval_predictions_total Total ML evaluation predictions",
            "# TYPE ml_eval_predictions_total counter",
            f"ml_eval_predictions_total {total}",
            "# HELP ml_eval_score_average Average ML evaluation score",
            "# TYPE ml_eval_score_average gauge",
            f"ml_eval_score_average {avg_score:.4f}",
        ]
        return "\n".join(lines) + "\n"
