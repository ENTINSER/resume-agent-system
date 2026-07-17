"""ML 评估服务

支持 Shadow / Online / Disabled 模式，可加载 production 模型进行推理，
并与 LLM 评估结果融合。
"""

from typing import Dict, Any, Optional, Tuple

from src.core.logger import logger
from src.core.config import settings
from src.core.metrics import observe_ml_disagreement
from src.ml.disagreement import DisagreementDetector
from src.ml.features import FeatureExtractor
from src.ml.model import EvalModel
from src.ml.registry import ModelRegistry, STAGE_PRODUCTION, STAGE_SHADOW


class MLEvaluationService:
    """ML 评估服务"""

    MODE_DISABLED = "disabled"
    MODE_SHADOW = "shadow"
    MODE_ONLINE = "online"

    def __init__(
        self,
        mode: Optional[str] = None,
        weight_ml: float = 0.3,
        model_id: Optional[str] = None,
        registry_path: Optional[str] = None,
    ):
        self.mode = mode or settings.ml_eval_mode
        self.weight_ml = weight_ml
        self.model_id = model_id
        self.registry = ModelRegistry(registry_path=registry_path) if registry_path else ModelRegistry()
        self._model: Optional[EvalModel] = None
        self._model_record: Optional[Dict[str, Any]] = None
        self._disagreement_detector = DisagreementDetector()
        self._load_model()

    def _load_model(self):
        """加载指定模型或 production 模型"""
        record = None
        if self.model_id:
            record = self.registry.get_model(self.model_id)
        if not record and self.mode != self.MODE_DISABLED:
            record = self.registry.get_production_model()

        if record:
            try:
                self._model = EvalModel.load(record["model_path"])
                self._model_record = record
                logger.info(f"[ML Service] 已加载模型 {record['id']}，stage={record['stage']}")
            except Exception as e:
                logger.warning(f"[ML Service] 加载模型失败: {e}")
                self._model = None
                self._model_record = None

    def is_available(self) -> bool:
        return self.mode != self.MODE_DISABLED and self._model is not None

    def predict(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """对 state 进行 ML 评估预测"""
        if not self.is_available():
            return None

        try:
            features = FeatureExtractor.extract(state)
            X = [FeatureExtractor.to_vector(features)]
            scores, passed = self._model.predict(X)
            return {
                "ml_score": float(scores[0]),
                "ml_passed": bool(passed[0]),
                "ml_features": features,
                "ml_model_id": self._model_record.get("id") if self._model_record else None,
                "ml_mode": self.mode,
            }
        except Exception as e:
            logger.warning(f"[ML Service] 预测失败: {e}")
            return None

    def blend_with_llm(
        self,
        llm_score: float,
        ml_score: float,
        llm_passed: bool,
        ml_passed: bool,
    ) -> Tuple[float, bool, str]:
        """融合 LLM 与 ML 评估结果

        返回: (final_score, final_passed, source)
        """
        if self.mode == self.MODE_SHADOW:
            # Shadow 模式不改变决策，仅记录 ml_score
            return llm_score, llm_passed, "llm"

        if self.mode == self.MODE_ONLINE:
            final_score = (1 - self.weight_ml) * llm_score + self.weight_ml * ml_score
            # 通过性采用“或”逻辑，但测试未通过时强制不通过由上层处理
            final_passed = llm_passed or ml_passed
            return round(final_score, 1), final_passed, "hybrid"

        return llm_score, llm_passed, "llm"

    def evaluate(self, state: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
        """完整评估入口：预测 ML 分数，并按模式与 LLM 结果融合"""
        prediction = self.predict(state)
        result = dict(llm_result)

        if prediction:
            result["ml_score"] = prediction["ml_score"]
            result["ml_passed"] = prediction["ml_passed"]
            result["ml_features"] = prediction["ml_features"]
            result["ml_model_id"] = prediction["ml_model_id"]
            result["ml_mode"] = prediction["ml_mode"]

            if self.mode == self.MODE_ONLINE:
                final_score, final_passed, source = self.blend_with_llm(
                    llm_score=llm_result.get("overall_score", 0),
                    ml_score=prediction["ml_score"],
                    llm_passed=llm_result.get("passed", False),
                    ml_passed=prediction["ml_passed"],
                )
                result["overall_score"] = final_score
                result["passed"] = final_passed
                result["source"] = source
                result["original_llm_score"] = llm_result.get("overall_score")
            elif self.mode == self.MODE_SHADOW:
                result["source"] = llm_result.get("source", "llm")

            disagreement = self._disagreement_detector.detect(llm_result, prediction)
            result["ml_disagreement_level"] = disagreement["level"]
            result["ml_disagreement_score_diff"] = disagreement["score_diff"]
            result["ml_disagreement_passed_mismatch"] = disagreement["passed_mismatch"]
            result["ml_disagreement_action"] = disagreement["action"]
            result["ml_disagreement_reason"] = disagreement["message"]
            observe_ml_disagreement(level=disagreement["level"])

            if disagreement["level"] == "severe":
                resolution = self._disagreement_detector.resolve(llm_result, prediction, disagreement)
                result["overall_score"] = resolution["overall_score"]
                result["passed"] = resolution["passed"]
                result["source"] = resolution["source"]
                logger.warning(
                    f"[ML Service] LLM/ML 严重分歧: {disagreement['message']} "
                    f"resolution={resolution}"
                )
        else:
            result["source"] = llm_result.get("source", "llm")

        return result
