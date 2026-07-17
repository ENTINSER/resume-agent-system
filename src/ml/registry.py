"""模型仓库 - 管理模型元数据、stage 与版本

本地使用 JSON 文件实现，接口设计向 MLflow Model Registry 对齐，
方便后续替换为真正的 MLflow / 自研 Registry。
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.core.logger import logger


DEFAULT_REGISTRY_PATH = "data/ml_model_registry.json"

# stage 定义与 MLflow 对齐
STAGE_SHADOW = "shadow"
STAGE_STAGING = "staging"
STAGE_PRODUCTION = "production"
STAGE_ARCHIVED = "archived"
VALID_STAGES = {STAGE_SHADOW, STAGE_STAGING, STAGE_PRODUCTION, STAGE_ARCHIVED}


class ModelRegistry:
    """模型注册表"""

    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.registry_path.exists() or self.registry_path.stat().st_size == 0:
            return {"version": "1.0", "models": []}
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def register(
        self,
        experiment_id: str,
        model_path: str,
        metrics: Dict[str, Any],
        stage: str = STAGE_STAGING,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """注册一个新模型版本"""
        if stage not in VALID_STAGES:
            raise ValueError(f"非法 stage: {stage}，可选: {VALID_STAGES}")

        # 如果注册为 production，把其他 production 降级为 staging
        if stage == STAGE_PRODUCTION:
            for m in self._data["models"]:
                if m.get("stage") == STAGE_PRODUCTION:
                    m["stage"] = STAGE_STAGING

        model_record = {
            "id": experiment_id,
            "model_path": os.path.abspath(model_path),
            "stage": stage,
            "metrics": metrics,
            "tags": tags or {},
            "registered_at": datetime.now().isoformat(),
        }
        self._data["models"].append(model_record)
        self._save()
        logger.info(f"[ML Registry] 注册模型 {experiment_id}，stage={stage}")
        return experiment_id

    def list_models(self, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出模型，可按 stage 过滤"""
        models = self._data["models"]
        if stage:
            models = [m for m in models if m.get("stage") == stage]
        return sorted(models, key=lambda x: x.get("registered_at", ""), reverse=True)

    def get_model(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """按 ID 查询模型"""
        for m in self._data["models"]:
            if m.get("id") == experiment_id:
                return m
        return None

    def get_production_model(self) -> Optional[Dict[str, Any]]:
        """获取当前 production 模型"""
        models = self.list_models(stage=STAGE_PRODUCTION)
        return models[0] if models else None

    def promote(self, experiment_id: str, stage: str) -> bool:
        """升级/降级模型 stage"""
        if stage not in VALID_STAGES:
            raise ValueError(f"非法 stage: {stage}")

        model = self.get_model(experiment_id)
        if not model:
            logger.warning(f"[ML Registry] 模型 {experiment_id} 不存在")
            return False

        # 若提升为 production，降级其他 production
        if stage == STAGE_PRODUCTION:
            for m in self._data["models"]:
                if m.get("stage") == STAGE_PRODUCTION and m.get("id") != experiment_id:
                    m["stage"] = STAGE_STAGING

        model["stage"] = stage
        model["updated_at"] = datetime.now().isoformat()
        self._save()
        logger.info(f"[ML Registry] 模型 {experiment_id} 已设置为 {stage}")
        return True

    def delete(self, experiment_id: str) -> bool:
        """删除模型记录并清理目录"""
        model = self.get_model(experiment_id)
        if not model:
            return False

        try:
            model_path = model.get("model_path")
            if model_path and Path(model_path).exists():
                shutil.rmtree(model_path)
        except Exception as e:
            logger.warning(f"[ML Registry] 清理模型目录失败: {e}")

        self._data["models"] = [m for m in self._data["models"] if m.get("id") != experiment_id]
        self._save()
        return True
