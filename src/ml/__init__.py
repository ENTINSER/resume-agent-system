"""企业级 ML 自动化评估模块

提供特征工程、样本管理、模型训练、模型注册、评估服务与监控能力。
"""

from src.ml.features import FeatureExtractor, FEATURE_VERSION
from src.ml.dataset import EvaluationDataset
from src.ml.model import EvalModel, ModelTrainer
from src.ml.registry import ModelRegistry
from src.ml.service import MLEvaluationService

__all__ = [
    "FeatureExtractor",
    "FEATURE_VERSION",
    "EvaluationDataset",
    "EvalModel",
    "ModelTrainer",
    "ModelRegistry",
    "MLEvaluationService",
]
