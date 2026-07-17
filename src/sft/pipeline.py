"""SFT 数据飞轮流水线

串联采集、质量过滤、入库、注册、训练触发。
"""

from typing import Dict, Any, List

from src.core.logger import logger
from src.core.config import settings
from src.sft.collectors import SFTCollectorPipeline, SFTSample
from src.sft.quality import QualityGate, Deduplicator
from src.sft.active_learning import sample_for_role
from src.sft.dataset import SFTDataset
from src.sft.registry import SFTRegistry
from src.sft.trainer import SFTTrainer


class SFTFlywheel:
    """SFT 数据飞轮"""

    def __init__(
        self,
        enabled: bool = None,
        roles: List[str] = None,
        min_score: float = None,
    ):
        self.enabled = enabled if enabled is not None else settings.sft_enabled
        self.roles = roles or settings.sft_roles.split(",")
        self.min_score = min_score if min_score is not None else settings.sft_min_score
        self.min_samples = settings.sft_min_samples

        # Phase 2 topic 4: 主动学习与训练阈值
        self.active_learning_enabled = settings.sft_active_learning_enabled
        self.active_learning_subset_size = settings.sft_active_learning_subset_size
        self.per_role_min_samples = settings.sft_per_role_min_samples
        self.max_duplicate_ratio = settings.sft_max_duplicate_ratio
        self.min_avg_score = settings.sft_min_avg_score

        self.collectors = SFTCollectorPipeline(roles=self.roles)
        self.quality_gate = QualityGate(
            min_score=self.min_score,
            per_role_min_samples=self.per_role_min_samples,
            max_duplicate_ratio=self.max_duplicate_ratio,
        )
        self.registry = SFTRegistry()

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """处理一个完成的任务 state"""
        result = {"enabled": self.enabled, "roles": {}, "samples_total": 0}
        if not self.enabled:
            return result

        # 采集
        samples = self.collectors.collect(state)
        if not samples:
            logger.info("[SFT Flywheel] 无可用样本")
            return result

        # 质量过滤
        filtered = self.quality_gate.filter_samples(state, samples)
        if not filtered:
            logger.info("[SFT Flywheel] 样本未通过质量门")
            return result

        # 去重
        filtered = Deduplicator.deduplicate(filtered)

        # 按角色入库；若启用主动学习且样本过多，则采样到子集大小
        for role in self.roles:
            role_samples = [s for s in filtered if s.role == role]
            if not role_samples:
                continue
            if (
                self.active_learning_enabled
                and len(role_samples) > self.active_learning_subset_size
            ):
                role_samples = sample_for_role(
                    role_samples, state, self.active_learning_subset_size
                )
            dataset = SFTDataset(role=role)
            added = dataset.add_samples(role_samples)
            stats = dataset.stats()
            self.registry.update(role, stats)
            result["roles"][role] = {
                "added": added,
                "total": stats["count"],
                "sampled": len(role_samples),
            }
            result["samples_total"] += added

        return result

    def maybe_train(self, role: str = None) -> Dict[str, Any]:
        """如果样本数与分布指标均达到阈值，触发训练"""
        result = {"trained": False}
        if not self.enabled:
            return result

        roles = [role] if role else self.roles
        for r in roles:
            dataset = SFTDataset(role=r)
            sample_dicts = dataset.load()
            samples = []
            for s in sample_dicts:
                fields = {k: v for k, v in s.items() if k in SFTSample.__dataclass_fields__}
                if "input" in s and "input_text" not in fields:
                    fields["input_text"] = s["input"]
                samples.append(SFTSample(**fields))
            distribution = self.quality_gate.check_distribution(samples)
            # 同步阈值，确保运行期配置变更生效
            self.quality_gate.per_role_min_samples = self.per_role_min_samples
            self.quality_gate.max_duplicate_ratio = self.max_duplicate_ratio
            validation = self.quality_gate.validate_distribution(distribution, r)

            if dataset.count() < self.per_role_min_samples:
                validation["reasons"].append(
                    f"count_below_threshold:{dataset.count()}<{self.per_role_min_samples}"
                )
                validation["passed"] = False
            if distribution["avg_score"] < self.min_avg_score:
                validation["reasons"].append(
                    f"avg_score_below_threshold:{distribution['avg_score']:.2f}<{self.min_avg_score}"
                )
                validation["passed"] = False

            if not validation["passed"]:
                logger.info(
                    f"[SFT Flywheel] {r} 不满足训练阈值，跳过训练: {validation['reasons']}"
                )
                continue

            trainer = SFTTrainer()
            artifacts = trainer.train(r, dataset)
            result = {
                "trained": True,
                "role": r,
                **artifacts,
            }
            logger.info(f"[SFT Flywheel] {r} 训练已触发: {artifacts['output_dir']}")
            # 每次只触发一个角色训练，避免资源争抢
            return result

        return result
