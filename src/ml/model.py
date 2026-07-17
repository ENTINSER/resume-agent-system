"""模型训练与评估

提供回归模型（预测 overall_score）和分类模型（预测 passed）。
"""

import os
import json
import uuid
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, accuracy_score, f1_score

from src.core.logger import logger
from src.ml.features import FeatureExtractor, FEATURE_VERSION


DEFAULT_MODEL_DIR = "models/ml_evaluator"


class EvalModel:
    """封装评分回归模型与通过分类模型"""

    def __init__(self, score_model=None, pass_model=None):
        self.score_model = score_model or GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self.pass_model = pass_model or RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42,
        )
        self.feature_version = FeatureExtractor.vector_columns()

    def fit(self, X: List[List[float]], y_score: List[float], y_passed: List[int]) -> None:
        """训练模型"""
        if not X:
            raise ValueError("训练数据为空")

        import numpy as np
        X_arr = np.array(X)
        y_score_arr = np.array(y_score)
        y_passed_arr = np.array(y_passed)

        logger.info(f"[ML Model] 开始训练，样本数: {len(X_arr)}")
        self.score_model.fit(X_arr, y_score_arr)
        self.pass_model.fit(X_arr, y_passed_arr)
        logger.info("[ML Model] 训练完成")

    def predict(self, X: List[List[float]]) -> Tuple[List[float], List[int]]:
        """预测评分与是否通过"""
        import numpy as np
        X_arr = np.array(X)
        scores = self.score_model.predict(X_arr).tolist()
        passed = self.pass_model.predict(X_arr).tolist()
        return scores, passed

    def feature_importances(self) -> Dict[str, float]:
        """返回特征重要性（基于评分模型）"""
        try:
            importances = self.score_model.feature_importances_
            columns = FeatureExtractor.vector_columns()
            return {columns[i]: float(importances[i]) for i in range(len(columns))}
        except Exception:
            return {}

    def save(self, model_dir: str) -> str:
        """保存模型到指定目录"""
        path = Path(model_dir)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "score_model.pkl", "wb") as f:
            pickle.dump(self.score_model, f)
        with open(path / "pass_model.pkl", "wb") as f:
            pickle.dump(self.pass_model, f)
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "feature_version": FeatureExtractor.vector_columns(),
                    "saved_at": datetime.now().isoformat(),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"[ML Model] 模型已保存: {model_dir}")
        return str(model_dir)

    @classmethod
    def load(cls, model_dir: str) -> "EvalModel":
        """从目录加载模型"""
        path = Path(model_dir)
        with open(path / "score_model.pkl", "rb") as f:
            score_model = pickle.load(f)
        with open(path / "pass_model.pkl", "rb") as f:
            pass_model = pickle.load(f)
        return cls(score_model=score_model, pass_model=pass_model)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def train(
        self,
        dataset,
        experiment_name: str = "ml_eval",
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """训练模型并返回实验报告"""
        X_train, y_score_train, y_passed_train, X_val, y_score_val, y_passed_val = dataset.train_test_split(
            test_size=test_size
        )

        if len(X_train) < 5:
            raise ValueError(f"训练样本不足: {len(X_train)}，至少需要 5 条")

        model = EvalModel()
        model.fit(X_train, y_score_train, y_passed_train)

        metrics = {}
        # 训练集指标
        train_scores, train_passed = model.predict(X_train)
        metrics["train"] = {
            "mae": mean_absolute_error(y_score_train, train_scores),
            "rmse": root_mean_squared_error(y_score_train, train_scores),
            "accuracy": accuracy_score(y_passed_train, train_passed),
            "f1": f1_score(y_passed_train, train_passed, zero_division=0),
            "samples": len(X_train),
        }

        # 验证集指标
        if X_val:
            val_scores, val_passed = model.predict(X_val)
            metrics["val"] = {
                "mae": mean_absolute_error(y_score_val, val_scores),
                "rmse": root_mean_squared_error(y_score_val, val_scores),
                "accuracy": accuracy_score(y_passed_val, val_passed),
                "f1": f1_score(y_passed_val, val_passed, zero_division=0),
                "samples": len(X_val),
            }

        # 保存模型
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        model_path = self.model_dir / experiment_id
        model.save(str(model_path))

        report = {
            "experiment_id": experiment_id,
            "model_path": str(model_path),
            "feature_version": FEATURE_VERSION,
            "metrics": metrics,
            "feature_importance": model.feature_importances(),
            "created_at": datetime.now().isoformat(),
        }

        with open(model_path / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"[ML Trainer] 实验 {experiment_id} 完成，验证 MAE: {metrics.get('val', {}).get('mae', 'N/A')}")
        return report
