"""编排器回调：阶段事件发布与取消信号检查"""

from typing import TYPE_CHECKING

from src.core.logger import logger
from src.core.exceptions import TaskCancelledError

if TYPE_CHECKING:
    from src.tasks.task_store import TaskStore


def publish_phase(
    task_store: "TaskStore",
    task_id: str,
    phase: str,
    state: dict,
) -> None:
    """向 Redis 发布阶段变更事件"""
    try:
        task_store.publish_phase(
            task_id,
            phase,
            payload={
                "status": state.get("status", ""),
                "current_phase": phase,
                "iteration_count": state.get("iteration_count", 0),
                "project_name": state.get("project", {}).get("name", ""),
            },
        )
    except Exception as e:
        logger.warning(f"[D] 发布阶段事件失败: {e}")


def check_cancel_signal(
    task_store: "TaskStore",
    task_id: str,
) -> None:
    """检查用户是否通过 Go API 发起取消"""
    try:
        if task_store.client.exists(f"agent:cancel:{task_id}"):
            logger.warning(f"[D] 任务 {task_id[:8]} 收到取消信号，正在终止")
            raise TaskCancelledError(f"任务 {task_id} 已被取消")
    except TaskCancelledError:
        raise
    except Exception as e:
        logger.warning(f"[D] 检查取消信号失败: {e}")
