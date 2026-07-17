"""LLM/ML 评估分歧检测与裁决

比较 LLM 与 ML 模型的评估结果，判定分歧等级并给出处理建议。
"""

from enum import Enum
from typing import Dict, Any

from src.core.config import settings


class DisagreementLevel(str, Enum):
    """分歧等级"""

    NONE = "none"
    MILD = "mild"
    SEVERE = "severe"


class DisagreementDetector:
    """LLM/ML 评估分歧检测器"""

    def __init__(
        self,
        threshold: float = None,
        severe_threshold: float = None,
    ):
        self.threshold = threshold if threshold is not None else settings.ml_eval_disagreement_threshold
        self.severe_threshold = severe_threshold if severe_threshold is not None else settings.ml_eval_severe_disagreement_threshold

    def detect(self, llm_result: Dict[str, Any], ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """检测 LLM 与 ML 评估结果之间的分歧

        Returns:
            {
                "level": "none" | "mild" | "severe",
                "score_diff": float,
                "passed_mismatch": bool,
                "message": str,
                "action": "accept" | "re_evaluate" | "human_review",
            }
        """
        llm_score = float(llm_result.get("overall_score", 0))
        ml_score = float(ml_result.get("ml_score", ml_result.get("overall_score", 0)))
        llm_passed = bool(llm_result.get("passed", False))
        ml_passed = bool(ml_result.get("ml_passed", False))

        score_diff = abs(llm_score - ml_score)
        passed_mismatch = llm_passed != ml_passed

        if passed_mismatch:
            level = DisagreementLevel.SEVERE
        elif score_diff >= self.severe_threshold:
            level = DisagreementLevel.SEVERE
        elif score_diff >= self.threshold:
            level = DisagreementLevel.MILD
        else:
            level = DisagreementLevel.NONE

        if level == DisagreementLevel.NONE:
            action = "accept"
            message = "LLM 与 ML 评估结果一致，接受当前结果。"
        elif level == DisagreementLevel.MILD:
            action = "re_evaluate"
            message = f"LLM/ML 分数差异 {score_diff:.1f} 达到轻度分歧阈值，建议重新评估。"
        else:
            action = "human_review"
            if passed_mismatch:
                message = f"LLM/ML 通过性不一致（LLM={llm_passed}, ML={ml_passed}），触发人工复核。"
            else:
                message = f"LLM/ML 分数差异 {score_diff:.1f} 达到严重分歧阈值，触发人工复核。"

        return {
            "level": level.value,
            "score_diff": round(score_diff, 1),
            "passed_mismatch": passed_mismatch,
            "message": message,
            "action": action,
        }

    def resolve(self, llm_result: Dict[str, Any], ml_result: Dict[str, Any], disagreement: Dict[str, Any]) -> Dict[str, Any]:
        """根据分歧等级裁决最终评估结果

        Returns:
            {
                "overall_score": float,
                "passed": bool,
                "source": "llm" | "ml" | "hybrid",
                "reason": str,
            }
        """
        level = disagreement.get("level", DisagreementLevel.NONE.value)
        llm_score = float(llm_result.get("overall_score", 0))
        ml_score = float(ml_result.get("ml_score", ml_result.get("overall_score", 0)))
        llm_passed = bool(llm_result.get("passed", False))
        ml_passed = bool(ml_result.get("ml_passed", False))

        if level == DisagreementLevel.NONE.value:
            return {
                "overall_score": llm_score,
                "passed": llm_passed,
                "source": "llm",
                "reason": "无分歧，采用 LLM 评估结果。",
            }

        if level == DisagreementLevel.MILD.value:
            avg_score = round((llm_score + ml_score) / 2, 1)
            return {
                "overall_score": avg_score,
                "passed": llm_passed and ml_passed,
                "source": "hybrid",
                "reason": f"轻度分歧，取 LLM/ML 平均分 {avg_score}，保守通过。",
            }

        # Severe: trust LLM as authoritative evaluator today
        return {
            "overall_score": llm_score,
            "passed": llm_passed,
            "source": "llm",
            "reason": f"严重分歧：{disagreement.get('message', '')} 当前以 LLM 为权威评估。",
        }
