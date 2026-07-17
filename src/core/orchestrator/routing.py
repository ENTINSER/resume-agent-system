"""编排器条件路由函数

所有路由函数只依赖 state，不依赖 Orchestrator 实例，因此使用 staticmethod 挂回类上。
"""

from typing import Literal

from src.core.shared_state import SharedState


def route_after_analysis(state: SharedState) -> Literal["develop", "human_review"]:
    """分析后路由"""
    if state.get("error"):
        return "human_review"
    return "develop"


def route_after_development(
    state: SharedState,
) -> Literal["evaluate", "human_review"]:
    """开发后路由：必要时转人工审核"""
    if state.get("human_review_required"):
        return "human_review"
    return "evaluate"


def route_after_evaluation(
    state: SharedState,
) -> Literal["train", "human_review", "rewrite", "end"]:
    """评估后路由"""
    eval_result = state.get("evaluation_result")

    if not eval_result:
        return "rewrite"

    if eval_result.get("passed"):
        if "模型训练" in state.get("requirements", "") or "微调" in state.get("requirements", ""):
            return "train"
        return "human_review"

    if state.get("iteration_count", 0) >= state.get("max_iterations", 10):
        return "end"

    return "rewrite"


def route_after_human(
    state: SharedState,
) -> Literal["continue", "train", "end", "evaluate"]:
    """人类审核后路由"""
    if state.get("human_approved"):
        return "end"

    feedback = state.get("human_feedback", "").lower()

    if "训练" in feedback or "train" in feedback:
        return "train"
    if "重写" in feedback or "rewrite" in feedback or "重新" in feedback:
        return "continue"
    if "评估" in feedback or "evaluate" in feedback:
        return "evaluate"

    return "continue"
