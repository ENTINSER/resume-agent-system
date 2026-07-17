"""ML 模型训练 Celery Task

任务完成后自动触发评估模型训练，形成数据飞轮。
"""

from src.tasks.celery_app import app
from src.core.logger import logger
from src.core.config import settings
from src.ml.dataset import EvaluationDataset
from src.ml.model import ModelTrainer
from src.ml.registry import ModelRegistry, STAGE_SHADOW


@app.task(bind=True, name="src.tasks.ml_training_task.train_ml_evaluator")
def train_ml_evaluator(
    self,
    min_samples: int = None,
    stage: str = None,
    experiment_name: str = "auto_ml_eval",
):
    """异步训练 ML 评估模型

    参数优先使用传入值，否则使用环境变量/默认值。
    """
    min_samples = min_samples or settings.ml_eval_min_samples
    stage = stage or settings.ml_eval_auto_stage

    dataset = EvaluationDataset()
    count = dataset.count()
    logger.info(f"[ML Training Task] 当前样本数: {count}，最低要求: {min_samples}")

    if count < min_samples:
        logger.info("[ML Training Task] 样本不足，跳过训练")
        return {"trained": False, "reason": "insufficient_samples", "count": count}

    try:
        trainer = ModelTrainer()
        report = trainer.train(dataset, experiment_name=experiment_name)

        registry = ModelRegistry()
        registry.register(
            experiment_id=report["experiment_id"],
            model_path=report["model_path"],
            metrics=report["metrics"],
            stage=stage,
            tags={"experiment_name": experiment_name, "trigger": "auto"},
        )

        logger.info(f"[ML Training Task] 训练完成: {report['experiment_id']}")
        return {
            "trained": True,
            "experiment_id": report["experiment_id"],
            "model_path": report["model_path"],
            "metrics": report["metrics"],
            "stage": stage,
        }
    except Exception as e:
        logger.error(f"[ML Training Task] 训练失败: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=2)
