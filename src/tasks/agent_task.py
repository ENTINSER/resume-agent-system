"""Agent 执行 Celery Task

Go API 通过 `run_agent_task.delay(task_id, requirements)` 投递任务，
Python Worker 消费并执行完整的 D-B-E-C 编排流程。
"""

import os
import traceback
from typing import Any, Dict, Optional

from src.tasks.celery_app import app
from src.tasks.task_store import TaskStore
from src.core.D import Orchestrator
from src.core.logger import logger
from src.core.config import settings
from src.core.exceptions import TransientError, TaskCancelledError, classify_exception
from src.core import metrics
from src.core.B import save_project
from src.vector_store.indexer import index_task_result


def _extract_target_dir(requirements: str) -> Optional[str]:
    """从 requirements 中提取用户指定的项目目录名，例如 'projects/test-calculator-v1'"""
    import re
    match = re.search(r"projects/([a-zA-Z0-9_\-]+)", requirements)
    return match.group(1) if match else None


# 全局 TaskStore 实例（每个 Worker 进程一个）
_task_store = TaskStore(redis_url=settings.redis_url)

# 启动 Prometheus 指标服务
metrics.start_metrics_server()


def _maybe_trigger_ml_training():
    """任务完成后异步触发 ML 模型训练"""
    try:
        from src.tasks.ml_training_task import train_ml_evaluator
        train_ml_evaluator.delay()
    except Exception as e:
        logger.warning(f"[Task] 触发 ML 训练任务失败: {e}")


def _maybe_collect_sft_samples(state: Dict[str, Any]):
    """任务完成后通过多指标质量门，只有批准后才触发 SFT 样本采集"""
    try:
        from src.core.quality_gates import MultiMetricGate
        gate = MultiMetricGate()
        gate_result = gate.check(state)
        if not gate_result["approved"]:
            logger.info(
                f"[Task] SFT 样本采集被质量门拦截: {gate_result['reasons']}"
            )
            return

        from src.sft.pipeline import SFTFlywheel
        flywheel = SFTFlywheel()
        sft_result = flywheel.process(state)
        state["sft_samples"] = state.get("sft_samples", []) + [sft_result]
        state["sft_dataset_saved"] = sft_result.get("samples_total", 0) > 0

        # 如果达到阈值，异步触发 SFT 训练
        train_result = flywheel.maybe_train()
        if train_result.get("trained"):
            state["sft_training_jobs"] = state.get("sft_training_jobs", []) + [train_result]
    except Exception as e:
        logger.warning(f"[Task] SFT 数据飞轮失败: {e}")



class TaskStateCallback:
    """编排器状态回调，把阶段变更写入 Redis 并发布事件"""

    def __init__(self, task_id: str, task_store: TaskStore):
        self.task_id = task_id
        self.task_store = task_store

    def on_phase_change(self, phase: str, payload: Dict[str, Any] = None) -> None:
        self.task_store.update_task(
            self.task_id,
            status="running",
            current_phase=phase,
        )
        self.task_store.publish_phase(self.task_id, phase, payload)


@app.task(bind=True, name="src.tasks.agent_task.run_agent_task")
def run_agent_task(self, task_id: str, requirements: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Celery Task：运行完整 Agent 工作流（支持幂等）"""
    task_store = _task_store

    # 幂等检查：若任务已在运行或已完成，直接返回现有状态
    existing = task_store.get_task(task_id)
    if existing:
        existing_status = existing.get("status", "")
        if existing_status == "completed" and existing.get("result"):
            logger.info(f"[Task] 任务 {task_id[:8]} 已完成，跳过重复执行")
            return existing["result"]
        if existing_status == "running":
            logger.info(f"[Task] 任务 {task_id[:8]} 正在运行，跳过重复执行")
            return {
                "task_id": task_id,
                "status": "running",
                "message": "任务正在运行中",
            }

    task_store.update_task(task_id, status="running", current_phase="init")

    try:
        # 创建编排器，UI 模式可通过环境变量 AGENT_UI_MODE 控制
        # 全自动化迭代时设置为 false，避免 30 分钟等待
        ui_mode = os.getenv("AGENT_UI_MODE", "true").lower() in ("true", "1", "yes")
        orchestrator = Orchestrator(ui_mode=ui_mode, task_id=task_id)

        # 运行工作流
        result = orchestrator.run(requirements)

        # 将结果索引到向量数据库（RAG 记忆）
        try:
            index_task_result(result)
        except Exception as e:
            logger.warning(f"[Task] 向量索引失败: {e}")

        # 触发 ML 评估模型训练（数据飞轮）
        eval_result = result.get("evaluation_result", {}) or {}
        passed = bool(eval_result.get("passed", False))
        try:
            metrics.observe_c_sample_candidate(passed=passed)
        except Exception as e:
            logger.warning(f"[Task] C 阶段指标记录失败: {e}")

        try:
            _maybe_trigger_ml_training()
        except Exception as e:
            logger.warning(f"[Task] 触发 ML 训练任务失败: {e}")

        # SFT 数据飞轮：采集高质量样本
        try:
            _maybe_collect_sft_samples(result)
        except Exception as e:
            logger.warning(f"[Task] SFT 数据飞轮失败: {e}")

        # 提取关键结果
        summary = {
            "session_id": result.get("session_id"),
            "status": result.get("status"),
            "current_phase": result.get("current_phase"),
            "project": result.get("project", {}),
            "iteration_count": result.get("iteration_count"),
            "evaluation_result": result.get("evaluation_result"),
            "total_tokens": result.get("total_tokens"),
            "total_cost": result.get("total_cost"),
            "total_duration_ms": result.get("total_duration_ms"),
        }

        task_store.update_task(
            task_id,
            status="completed",
            current_phase=result.get("current_phase", "done"),
            result=summary,
        )

        # 把生成的代码落盘到 projects/，方便验收
        if result.get("status") in ("complete", "completed"):
            try:
                target_dir = _extract_target_dir(requirements)
                saved_dir = save_project(result, "projects", project_dir_hint=target_dir)
                logger.info(f"[Task] 项目已保存到: {saved_dir}")
            except Exception as e:
                logger.warning(f"[Task] 保存项目文件失败: {e}")

        # 按 API Key 累计用量（配额与审计）
        if api_key:
            try:
                task_store.increment_quota(
                    api_key,
                    int(summary.get("total_tokens") or 0),
                    float(summary.get("total_cost") or 0.0),
                )
            except Exception as e:
                logger.warning(f"[Task] 用量统计失败: {e}")

        # Prometheus 指标
        try:
            duration_ms = float(summary.get("total_duration_ms") or 0)
            metrics.observe_task(
                status=summary.get("status", "unknown"),
                duration_seconds=duration_ms / 1000.0,
            )
            metrics.observe_llm(
                phase="total",
                tokens=int(summary.get("total_tokens") or 0),
                cost=float(summary.get("total_cost") or 0.0),
            )
        except Exception as e:
            logger.warning(f"[Task] 指标上报失败: {e}")

        return summary

    except TaskCancelledError as e:
        logger.warning(f"[Task] 任务 {task_id[:8]} 已取消")
        task_store.update_task(
            task_id,
            status="cancelled",
            current_phase="cancelled",
            error=str(e),
        )
        return {"task_id": task_id, "status": "cancelled"}

    except Exception as e:
        classified = classify_exception(e)
        error_type = type(classified).__name__
        error_msg = f"{classified}\n{traceback.format_exc()}"

        if isinstance(classified, TransientError) and self.request.retries < 3:
            logger.warning(
                f"[Task] 任务 {task_id[:8]} 遇到临时错误，{self.request.retries}/3 秒后重试: {classified}",
                extra={"error_type": error_type, "retry_count": self.request.retries},
            )
            countdown = min(2 ** self.request.retries, 60)
            raise self.retry(exc=classified, countdown=countdown)

        logger.error(
            f"[Task] 任务 {task_id[:8]} 最终失败 [{error_type}]: {classified}",
            extra={"error_type": error_type},
        )
        task_store.update_task(
            task_id,
            status="failed",
            current_phase="error",
            error=error_msg,
        )
        raise classified
