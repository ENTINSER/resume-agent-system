"""SFT 数据集仓库

支持多角色独立存储、版本快照、训练/验证拆分。
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.logger import logger
from src.sft.collectors import SFTSample
from src.sft.quality import Deduplicator


DEFAULT_SFT_DIR = "data/sft_datasets"
DEFAULT_VERSION = "v1"


class SFTDataset:
    """单个角色的 SFT 数据集"""

    def __init__(
        self,
        role: str,
        dataset_dir: str = DEFAULT_SFT_DIR,
        version: str = DEFAULT_VERSION,
    ):
        self.role = role
        self.version = version
        self.role_dir = Path(dataset_dir) / role / version
        self.dataset_path = self.role_dir / "dataset.jsonl"
        self.role_dir.mkdir(parents=True, exist_ok=True)

    def _hash(self, sample: SFTSample) -> str:
        return Deduplicator.hash_sample(sample)

    def _load_existing_ids(self) -> set:
        ids = set()
        if not self.dataset_path.exists():
            return ids
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return ids

    def add_samples(self, samples: List[SFTSample]) -> int:
        """追加样本，返回新增数量"""
        existing_ids = self._load_existing_ids()
        added = 0

        with open(self.dataset_path, "a", encoding="utf-8") as f:
            for sample in samples:
                if sample.role != self.role:
                    continue
                sample_id = self._hash(sample)
                if sample_id in existing_ids:
                    continue
                record = sample.to_dict()
                record["id"] = sample_id
                record["stored_at"] = datetime.now().isoformat()
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                existing_ids.add(sample_id)
                added += 1

        logger.info(f"[SFT Dataset] {self.role} 新增 {added} 条样本")
        return added

    def load(self) -> List[Dict[str, Any]]:
        """加载全部样本"""
        samples = []
        if not self.dataset_path.exists():
            return samples
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return samples

    def split(self, test_size: float = 0.1, random_state: int = 42) -> Dict[str, Any]:
        """拆分训练/验证集"""
        from sklearn.model_selection import train_test_split

        samples = self.load()
        if len(samples) < 5:
            return {
                "train": samples,
                "val": [],
                "train_count": len(samples),
                "val_count": 0,
            }

        train, val = train_test_split(
            samples,
            test_size=test_size,
            random_state=random_state,
        )
        return {
            "train": train,
            "val": val,
            "train_count": len(train),
            "val_count": len(val),
        }

    def snapshot(self) -> str:
        """创建版本快照"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.role_dir / f"snapshot_{timestamp}.jsonl"
        samples = self.load()
        with open(snapshot_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info(f"[SFT Dataset] {self.role} 快照已保存: {snapshot_path}")
        return str(snapshot_path)

    def count(self) -> int:
        return len(self.load())

    def stats(self) -> Dict[str, Any]:
        samples = self.load()
        scores = [s.get("quality_score", 0) for s in samples]
        return {
            "role": self.role,
            "version": self.version,
            "count": len(samples),
            "avg_quality_score": sum(scores) / len(scores) if scores else 0,
            "min_quality_score": min(scores) if scores else 0,
            "max_quality_score": max(scores) if scores else 0,
        }
