"""编排器包

对外的 Orchestrator 类与全局实例仍通过 ``src.core.D`` 访问，
这里提供懒加载单例，避免导入时立即实例化 SQLite / RAG。
"""

from typing import Optional

from src.core.orchestrator.core import Orchestrator

__all__ = ["Orchestrator", "get_orchestrator", "orchestrator"]


_orchestrator_instance: Optional[Orchestrator] = None


def get_orchestrator(ui_mode: bool = False, task_id: Optional[str] = None) -> Orchestrator:
    """获取全局 Orchestrator 实例（进程内单例）"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = Orchestrator(ui_mode=ui_mode, task_id=task_id)
    return _orchestrator_instance


class _LazyOrchestrator:
    """懒加载代理：导入 ``src.core.D`` 时不会创建真实 Orchestrator"""

    def _get(self) -> Orchestrator:
        return get_orchestrator()

    def __getattr__(self, name: str):
        return getattr(self._get(), name)

    def __call__(self, *args, **kwargs):
        return self._get()(*args, **kwargs)


orchestrator = _LazyOrchestrator()
