"""样本仓库 - 评估数据集管理

数据以 JSONL 形式存储，支持版本快照、训练/测试拆分与去重。
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from src.core.logger import logger
from src.ml.features import FeatureExtractor


DEFAULT_DATASET_DIR = "data/ml_eval_dataset"
DEFAULT_DATASET_VERSION = "v1"


class EvaluationDataset:
    """评估样本仓库"""

    def __init__(
        self,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        version: str = DEFAULT_DATASET_VERSION,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.version = version
        self.version_dir = self.dataset_dir / version
        self.dataset_path = self.version_dir / "dataset.jsonl"
        self.version_dir.mkdir(parents=True, exist_ok=True)

    def _compute_sample_hash(self, features: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """基于特征与元数据计算样本哈希，用于去重"""
        content = json.dumps({"features": features, "metadata": metadata}, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def append(
        self,
        state: Dict[str, Any],
        evaluation_result: Dict[str, Any],
    ) -> bool:
        """追加一条样本

        返回是否成功追加（False 表示重复或无效）。
        """
        try:
            features = FeatureExtractor.extract(state)
        except Exception as e:
            logger.warning(f"[ML Dataset] 特征提取失败: {e}")
            return False

        overall_score = evaluation_result.get("overall_score")
        passed = evaluation_result.get("passed")
        if overall_score is None or passed is None:
            logger.warning("[ML Dataset] 缺少标签，跳过")
            return False

        stable_metadata = {
            "session_id": state.get("session_id", ""),
            "task_id": state.get("task_id", ""),
            "source": evaluation_result.get("source", "unknown"),
        }
        sample_hash = self._compute_sample_hash(features, stable_metadata)

        metadata = {
            **stable_metadata,
            "timestamp": datetime.now().isoformat(),
        }

        sample = {
            "id": sample_hash,
            "features": features,
            "labels": {
                "overall_score": float(overall_score),
                "passed": bool(passed),
                "dimensions": evaluation_result.get("dimensions", {}),
            },
            "metadata": metadata,
        }

        # 简单去重：基于 id
        existing_ids = set()
        if self.dataset_path.exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            existing_ids.add(json.loads(line)["id"])
                        except (json.JSONDecodeError, KeyError):
                            continue

        if sample_hash in existing_ids:
            logger.info("[ML Dataset] 样本已存在，跳过")
            return False

        with open(self.dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        logger.info(f"[ML Dataset] 已追加样本 {sample_hash}")
        return True

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

    def get_xy(self) -> Tuple[List[List[float]], List[float], List[int]]:
        """获取训练用 X, y_score, y_passed"""
        samples = self.load()
        X = []
        y_score = []
        y_passed = []
        for sample in samples:
            features = sample.get("features", {})
            labels = sample.get("labels", {})
            try:
                X.append(FeatureExtractor.to_vector(features))
                y_score.append(float(labels.get("overall_score", 0)))
                y_passed.append(1 if labels.get("passed", False) else 0)
            except Exception as e:
                logger.warning(f"[ML Dataset] 样本解析失败: {e}")
                continue
        return X, y_score, y_passed

    def train_test_split(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[List[List[float]], List[float], List[int], List[List[float]], List[float], List[int]]:
        """拆分训练集与验证集"""
        from sklearn.model_selection import train_test_split as sk_split

        X, y_score, y_passed = self.get_xy()
        if len(X) < 5:
            return X, y_score, y_passed, [], [], []

        X_train, X_val, y_score_train, y_score_val, y_passed_train, y_passed_val = sk_split(
            X,
            y_score,
            y_passed,
            test_size=test_size,
            random_state=random_state,
            stratify=None if len(set(y_passed)) < 2 else y_passed,
        )
        return X_train, y_score_train, y_passed_train, X_val, y_score_val, y_passed_val

    def version_snapshot(self) -> str:
        """对当前数据集做版本快照"""
        snapshot_path = self.version_dir / f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        samples = self.load()
        with open(snapshot_path, "w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        logger.info(f"[ML Dataset] 数据集快照已保存: {snapshot_path}")
        return str(snapshot_path)

    def count(self) -> int:
        """返回样本数量"""
        return len(self.load())
