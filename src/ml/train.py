"""ML 评估模型训练脚本

用法：
    source venv/bin/activate
    python -m src.ml.train --min-samples 20 --stage production
"""

import argparse
import os

from src.ml.dataset import EvaluationDataset
from src.ml.model import ModelTrainer
from src.ml.registry import ModelRegistry, STAGE_PRODUCTION
from src.core.logger import logger


def main():
    parser = argparse.ArgumentParser(description="训练 ML 评估模型")
    parser.add_argument("--min-samples", type=int, default=20, help="最小训练样本数")
    parser.add_argument("--stage", default=STAGE_PRODUCTION, help="注册 stage（shadow/staging/production）")
    parser.add_argument("--experiment-name", default="ml_eval", help="实验名称")
    args = parser.parse_args()

    dataset = EvaluationDataset()
    count = dataset.count()
    logger.info(f"[ML Train] 当前样本数: {count}")

    if count < args.min_samples:
        logger.warning(f"[ML Train] 样本不足 {args.min_samples}，停止训练")
        return

    trainer = ModelTrainer()
    report = trainer.train(dataset, experiment_name=args.experiment_name)

    registry = ModelRegistry()
    registry.register(
        experiment_id=report["experiment_id"],
        model_path=report["model_path"],
        metrics=report["metrics"],
        stage=args.stage,
        tags={"experiment_name": args.experiment_name},
    )

    logger.info(f"[ML Train] 训练完成，模型 ID: {report['experiment_id']}")
    logger.info(f"[ML Train] 验证 MAE: {report['metrics'].get('val', {}).get('mae', 'N/A')}")


if __name__ == "__main__":
    main()
