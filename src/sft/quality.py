"""SFT 数据质量中心

负责数据过滤、安全检查、去重。
"""

import re
import json
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import asdict

from src.core.logger import logger
from src.sft.collectors import SFTSample


# Secrets 检测模式
SECRET_PATTERNS = [
    (r"(?i)(api[_-]?key\s*[:=]\s*)['\"]?[a-z0-9_\-]{16,}['\"]?", "API_KEY"),
    (r"(?i)(password\s*[:=]\s*)['\"][^'\"]+['\"]", "PASSWORD"),
    (r"(?i)(secret\s*[:=]\s*)['\"]?[a-z0-9_\-]{16,}['\"]?", "SECRET"),
    (r"(?i)(token\s*[:=]\s*)['\"]?[a-z0-9_\-]{16,}['\"]?", "TOKEN"),
    (r"(?i)sk-[a-z0-9]{48}", "OPENAI_API_KEY"),
    (r"(?i)AK[A-Za-z0-9]{16,}", "ALIYUN_AK"),
]

# PII 检测模式
PII_PATTERNS = [
    (r"\b1[3-9]\d{9}\b", "PHONE"),  # 手机号
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "EMAIL"),
    (r"\b\d{17}[\dXx]\b", "ID_CARD"),  # 身份证号
]


class QualityGate:
    """SFT 数据质量门"""

    def __init__(
        self,
        min_score: float = 70.0,
        require_passed: bool = True,
        require_no_syntax_error: bool = True,
        require_all_tests_passed: bool = False,
        enable_secret_scan: bool = True,
        enable_pii_scan: bool = True,
        per_role_min_samples: Optional[int] = None,
        max_duplicate_ratio: Optional[float] = None,
    ):
        self.min_score = min_score
        self.require_passed = require_passed
        self.require_no_syntax_error = require_no_syntax_error
        self.require_all_tests_passed = require_all_tests_passed
        self.enable_secret_scan = enable_secret_scan
        self.enable_pii_scan = enable_pii_scan
        self.per_role_min_samples = per_role_min_samples
        self.max_duplicate_ratio = max_duplicate_ratio

    def check_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """检查整个任务 state 是否满足质量要求"""
        evaluation = state.get("evaluation_result", {}) or {}
        test_results = state.get("test_results", []) or []
        static_metrics = state.get("static_metrics", {}) or {}

        passed = evaluation.get("passed", False)
        score = float(evaluation.get("overall_score", 0))
        syntax_errors = len(static_metrics.get("syntax_errors", []) or [])
        all_tests_passed = all(r.get("passed", False) for r in test_results) if test_results else True

        reasons = []
        if self.require_passed and not passed:
            reasons.append("evaluation_not_passed")
        if score < self.min_score:
            reasons.append(f"score_below_threshold:{score}<{self.min_score}")
        if self.require_no_syntax_error and syntax_errors > 0:
            reasons.append(f"syntax_errors:{syntax_errors}")
        if self.require_all_tests_passed and not all_tests_passed:
            reasons.append("tests_not_all_passed")

        return {
            "approved": len(reasons) == 0,
            "reasons": reasons,
            "score": score,
            "passed": passed,
        }

    def check_sample(self, sample: SFTSample) -> Dict[str, Any]:
        """检查单个样本是否安全"""
        text = f"{sample.instruction}\n{sample.input_text}\n{sample.output}"

        issues = []
        if self.enable_secret_scan:
            for pattern, issue_type in SECRET_PATTERNS:
                if re.search(pattern, text):
                    issues.append(f"secret:{issue_type}")

        if self.enable_pii_scan:
            for pattern, issue_type in PII_PATTERNS:
                if re.search(pattern, text):
                    issues.append(f"pii:{issue_type}")

        return {
            "safe": len(issues) == 0,
            "issues": issues,
        }

    def filter_samples(self, state: Dict[str, Any], samples: List[SFTSample]) -> List[SFTSample]:
        """过滤样本：先检查 state 质量，再检查每个样本安全"""
        state_check = self.check_state(state)
        if not state_check["approved"]:
            logger.info(f"[SFT Quality] State 未通过质量门: {state_check['reasons']}")
            return []

        approved = []
        for sample in samples:
            sample_check = self.check_sample(sample)
            if sample_check["safe"]:
                approved.append(sample)
            else:
                logger.info(f"[SFT Quality] 样本不安全，跳过: {sample_check['issues']}")
        return approved

    def check_distribution(self, samples: List[SFTSample]) -> Dict[str, Any]:
        """计算样本分布质量指标。

        - avg_score: 平均质量分（优先取 ``metadata.score``，否则回退到 ``quality_score``）。
        - duplicate_ratio: 去重前后重复样本占比。
        - role_counts: 各角色样本数量。
        """
        if not samples:
            return {
                "avg_score": 0.0,
                "duplicate_ratio": 0.0,
                "role_counts": {},
            }

        total = len(samples)
        unique = Deduplicator.deduplicate(samples)
        duplicate_ratio = (total - len(unique)) / total

        scores = []
        role_counts: Dict[str, int] = {}
        for sample in samples:
            score = 0.0
            if hasattr(sample, "metadata") and sample.metadata:
                score = sample.metadata.get("score", 0.0)
            elif hasattr(sample, "context") and sample.context:
                score = sample.context.get("score", 0.0)
            elif hasattr(sample, "quality_score"):
                score = sample.quality_score
            scores.append(float(score))

            role_counts[sample.role] = role_counts.get(sample.role, 0) + 1

        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "avg_score": avg_score,
            "duplicate_ratio": duplicate_ratio,
            "role_counts": role_counts,
        }

    def validate_distribution(
        self,
        distribution: Dict[str, Any],
        role: str,
    ) -> Dict[str, Any]:
        """校验样本分布是否满足训练阈值。

        ``per_role_min_samples`` 与 ``max_duplicate_ratio`` 为 ``None`` 时不启用对应检查，
        保持向后兼容。
        """
        reasons = []
        if self.per_role_min_samples is not None:
            count = distribution.get("role_counts", {}).get(role, 0)
            if count < self.per_role_min_samples:
                reasons.append(
                    f"role_{role}_samples_below_threshold:{count}<{self.per_role_min_samples}"
                )

        if self.max_duplicate_ratio is not None:
            duplicate_ratio = distribution.get("duplicate_ratio", 0.0)
            if duplicate_ratio > self.max_duplicate_ratio:
                reasons.append(
                    f"duplicate_ratio_above_threshold:{duplicate_ratio:.4f}>{self.max_duplicate_ratio}"
                )

        return {
            "passed": len(reasons) == 0,
            "reasons": reasons,
        }


class Deduplicator:
    """基于哈希去重"""

    @staticmethod
    def hash_sample(sample: SFTSample) -> str:
        content = f"{sample.role}|{sample.instruction}|{sample.input_text}|{sample.output}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    @staticmethod
    def deduplicate(samples: List[SFTSample]) -> List[SFTSample]:
        seen = set()
        unique = []
        for sample in samples:
            h = Deduplicator.hash_sample(sample)
            if h not in seen:
                seen.add(h)
                unique.append(sample)
        return unique
