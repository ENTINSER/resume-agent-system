"""D - Orchestrator（编排调度中心）

本模块现在是一个薄 facade，实际实现已迁移到 ``src.core.orchestrator`` 包。
保留此文件是为了兼容现有 import 路径（``src.main``、``src.tasks.agent_task``、
``src.ui.web_ui``、测试等）。
"""

from src.core.orchestrator import Orchestrator, orchestrator, get_orchestrator

__all__ = ["Orchestrator", "orchestrator", "get_orchestrator"]
