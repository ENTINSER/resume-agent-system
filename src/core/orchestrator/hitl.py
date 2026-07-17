"""编排器人类审核（HITL）节点"""

import json
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING

from src.core.shared_state import SharedState
from src.core.logger import logger
from src.core.config import settings
from src.core.orchestrator.constants import PENDING_REVIEW_FILE, HUMAN_RESPONSE_FILE

if TYPE_CHECKING:
    from src.core.orchestrator.core import Orchestrator


def human_review_node(self, state: SharedState):
    """人类审核节点 - 支持CLI和UI两种模式"""
    state["status"] = "human_review"
    state["current_phase"] = "human_review"

    # 计算时间和成本
    llm_usage = state.get("llm_usage", [])
    total_tokens = sum(u.get("total_tokens", 0) for u in llm_usage)
    total_cost = sum(u.get("cost_usd", 0.0) for u in llm_usage)
    total_duration = sum(u.get("elapsed_ms", 0) for u in llm_usage)

    # 构建审核信息
    review_info = self._build_review_info(state, total_tokens, total_cost, total_duration)

    if self.ui_mode:
        feedback = self._wait_for_ui_response(state, review_info)
    else:
        feedback = self._wait_for_cli_input(state, review_info)

    state["human_approved"] = feedback == "approve"
    state["human_feedback"] = feedback

    self._save_state(state, "human_review")
    return state


def build_review_info(
    state: SharedState,
    total_tokens: int,
    total_cost: float,
    total_duration: int,
) -> dict:
    """构建审核信息字典（供CLI和UI共用）"""
    eval_result = state.get("evaluation_result")
    project = state.get("project")

    info = {
        "session_id": state.get("session_id", ""),
        "phase": state.get("current_phase", ""),
        "iteration": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 10),
        "project_name": project.get("name", "") if project else "",
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "total_duration_ms": total_duration,
        "code_artifacts": [a.get("file_path", "") for a in state.get("code_artifacts", [])],
        "overall_score": eval_result.get("overall_score", 0) if eval_result else 0,
        "passed": eval_result.get("passed", False) if eval_result else False,
        "training_jobs": state.get("training_jobs", []),
    }
    return info


def wait_for_ui_response(
    self,
    state: SharedState,
    review_info: dict,
) -> str:
    """UI模式：优先通过 Redis 等待用户响应，Redis 不可用时回退到文件轮询"""
    max_wait = settings.hitl_timeout_seconds
    timeout_action = settings.hitl_timeout_action
    logger.info(f"[D] 等待UI用户输入... 超时时间: {max_wait}s, 超时动作: {timeout_action}")

    # 写入 pending 文件（兼容旧 UI）
    pending_data = {
        "status": "pending",
        "timestamp": datetime.now().isoformat(),
        **review_info,
    }
    os.makedirs("data", exist_ok=True)
    with open(PENDING_REVIEW_FILE, "w") as f:
        json.dump(pending_data, f, ensure_ascii=False, indent=2)

    # 清理旧的 response 文件
    if os.path.exists(HUMAN_RESPONSE_FILE):
        os.remove(HUMAN_RESPONSE_FILE)

    # 通过 Redis 发布等待审核事件
    task_id = state.get("task_id")
    if self._task_store and task_id:
        self._task_store.update_task(
            task_id,
            status="human_review",
            current_phase="human_review",
        )
        review_payload = {"review_info": review_info}
        self._task_store.publish_phase(task_id, "human_review", review_payload)

    # 轮询等待用户响应（每2秒检查一次）
    waited = 0
    redis_key = f"agent:human_review:{task_id}" if task_id else None

    while waited < max_wait:
        # 1. 优先检查 Redis（Go API 写入）
        if self._task_store and redis_key:
            try:
                feedback = self._task_store.client.get(redis_key)
                if feedback:
                    feedback = feedback.decode("utf-8") if isinstance(feedback, bytes) else feedback
                    logger.info(f"[D] 收到Redis UI响应: {feedback}")
                    self._task_store.client.delete(redis_key)
                    _cleanup_review_files()
                    return feedback
            except Exception as e:
                logger.warning(f"[D] Redis 读取反馈失败: {e}")

        # 2. 回退检查文件（兼容旧 UI）
        if os.path.exists(HUMAN_RESPONSE_FILE):
            try:
                with open(HUMAN_RESPONSE_FILE, "r") as f:
                    response = json.load(f)
                feedback = response.get("feedback", "approve")
                logger.info(f"[D] 收到文件UI响应: {feedback}")
                _cleanup_review_files()
                return feedback
            except Exception as e:
                logger.error(f"[D] 读取UI响应失败: {e}")
                time.sleep(2)
                waited += 2
                continue

        time.sleep(2)
        waited += 2
        if waited % 60 == 0:
            logger.info(f"[D] 已等待 {waited // 60} 分钟...")

    # 超时，根据配置执行动作
    _cleanup_review_files()
    if timeout_action == "approve":
        logger.warning(f"[D] UI等待超时，执行配置动作: {timeout_action} -> approve")
        return "approve"
    elif timeout_action == "decline":
        logger.warning(f"[D] UI等待超时，执行配置动作: {timeout_action} -> rewrite")
        return "rewrite"
    else:  # abort
        from src.core.exceptions import TaskCancelledError

        logger.warning(f"[D] UI等待超时，执行配置动作: {timeout_action} -> 取消任务")
        raise TaskCancelledError("HITL 等待超时，任务已取消")


def wait_for_cli_input(self, state: SharedState, review_info: dict) -> str:
    """CLI模式：阻塞等待终端输入"""
    if settings.auto_approve:
        print("\n" + "=" * 70)
        print("🛑 Human-in-the-loop 检查点")
        print("=" * 70)
        print("AUTO_APPROVE=true，自动批准通过")
        print("=" * 70 + "\n")
        return "approve"

    print("\n" + "=" * 70)
    print("🛑 Human-in-the-loop 检查点")
    print("=" * 70)
    print(f"当前阶段: {review_info['phase']}")
    print(f"迭代次数: {review_info['iteration']}/{review_info['max_iterations']}")

    if review_info["project_name"]:
        print(f"项目名称: {review_info['project_name']}")

    if review_info["overall_score"] > 0:
        print(f"\n评估得分: {review_info['overall_score']}/100")
        print(f"是否通过: {'✅' if review_info['passed'] else '❌'}")

    print(f"\n代码产出: {len(review_info['code_artifacts'])} 个文件")
    for path in review_info["code_artifacts"][:5]:
        print(f"  - {path}")

    print(f"\n📊 成本统计:")
    print(f"  Tokens: {review_info['total_tokens']:,}")
    print(f"  成本: ${review_info['total_cost']:.4f}")
    print(f"  LLM耗时: {review_info['total_duration_ms']}ms")

    if review_info["training_jobs"]:
        job = review_info["training_jobs"][-1]
        print(f"\n🔧 训练任务:")
        print(f"  ID: {job.get('id', '')}")
        print(f"  状态: {job.get('status', '')}")
        print(f"  模型: {job.get('model_name', '')}")

    print("\n操作选项:")
    print("  [1] approve   批准通过")
    print("  [2] rewrite   要求重写")
    print("  [3] evaluate  重新评估")
    print("  [4] train     开始训练")
    print("=" * 70 + "\n")

    while True:
        try:
            feedback = input("请选择操作 (1-approve/2-rewrite/3-evaluate/4-train): ").strip().lower()
            if feedback in ("1", "approve", "通过", "a"):
                print("✅ 已批准")
                return "approve"
            elif feedback in ("2", "rewrite", "重写", "r"):
                print("🔄 要求重写")
                return "rewrite"
            elif feedback in ("3", "evaluate", "评估", "e"):
                print("📊 重新评估")
                return "evaluate"
            elif feedback in ("4", "train", "训练", "t"):
                print("🎯 开始训练")
                return "train"
            else:
                print("无效输入，请重新选择")
        except (EOFError, KeyboardInterrupt):
            print("\n收到中断信号，默认批准")
            return "approve"


def _cleanup_review_files() -> None:
    """清理审核相关文件"""
    for f in (PENDING_REVIEW_FILE, HUMAN_RESPONSE_FILE):
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception:
            pass
