"""SFT 数据集注册表

记录各角色数据集的版本、规模、质量分布。
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from src.core.logger import logger


DEFAULT_REGISTRY_PATH = "data/sft_registry.json"


class SFTRegistry:
    """SFT 数据集注册表"""

    def __init__(self, registry_path: str = DEFAULT_REGISTRY_PATH):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.registry_path.exists() or self.registry_path.stat().st_size == 0:
            return {"version": "1.0", "datasets": {}}
        with open(self.registry_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def update(self, role: str, stats: Dict[str, Any]) -> None:
        """更新某个角色的数据集元数据"""
        self._data["datasets"][role] = {
            **stats,
            "updated_at": datetime.now().isoformat(),
        }
        self._save()
        logger.info(f"[SFT Registry] 更新 {role} 数据集元数据")

    def get(self, role: str) -> Dict[str, Any]:
        return self._data["datasets"].get(role, {})

    def list_roles(self) -> List[str]:
        return list(self._data["datasets"].keys())

    def snapshot(self, role: str, snapshot_path: str) -> None:
        """记录快照路径"""
        if role not in self._data["datasets"]:
            self._data["datasets"][role] = {}
        self._data["datasets"][role]["latest_snapshot"] = snapshot_path
        self._data["datasets"][role]["snapshot_at"] = datetime.now().isoformat()
        self._save()
