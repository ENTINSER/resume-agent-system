"""Orchestrator 核心类

负责初始化、状态图构建、流程运行/恢复、状态保存与总结输出。
节点与路由函数拆分到同包的 nodes / routing / hitl 模块，再以类属性挂回。
"""

import os
import sqlite3
from datetime import datetime
from typing import Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core.shared_state import SharedState, create_initial_state
from src.core.logger import logger
from src.core.exceptions import (
    classify_exception,
    TransientError,
    DegradableError,
    FatalError,
)
from src.core.persistence import get_session_store
from src.core.config import settings
from src.core import metrics
from src.vector_store.rag_retriever import RAGRetriever

from src.core.orchestrator import callbacks
from src.core.orchestrator.nodes import analyze_node, analyze_with_llm, fallback_analysis
from src.core.orchestrator.routing import (
    route_after_analysis,
    route_after_development,
    route_after_evaluation,
    route_after_human,
)
from src.core.orchestrator.hitl import (
    human_review_node,
    build_review_info,
    wait_for_ui_response,
    wait_for_cli_input,
)


class Orchestrator:
    """编排器 - 管理4智能体协作流程"""

    # 将拆分出去的函数以同名挂回类上，保持外部调用与测试兼容
    _analyze_node = analyze_node
    _analyze_with_llm = analyze_with_llm
    _fallback_analysis = staticmethod(fallback_analysis)
    _route_after_analysis = staticmethod(route_after_analysis)
    _route_after_development = staticmethod(route_after_development)
    _route_after_evaluation = staticmethod(route_after_evaluation)
    _route_after_human = staticmethod(route_after_human)
    _human_review_node = human_review_node
    _build_review_info = staticmethod(build_review_info)
    _wait_for_ui_response = wait_for_ui_response
    _wait_for_cli_input = wait_for_cli_input

    def __init__(self, ui_mode: bool = False, task_id: Optional[str] = None):
        self.store = get_session_store()
        self.ui_mode = ui_mode
        self.task_id = task_id

        # LangGraph SqliteSaver checkpointer：实现断点续跑
        os.makedirs("data", exist_ok=True)
        self._checkpoint_conn = sqlite3.connect(
            "data/checkpoints.sqlite", check_same_thread=False
        )
        self.checkpointer = SqliteSaver(self._checkpoint_conn)

        self.workflow = self._build_graph()

        # 异步模式下通过 Redis 接收人类反馈
        self._task_store = None
        if task_id:
            try:
                from src.tasks.task_store import TaskStore

                self._task_store = TaskStore()
            except Exception as e:
                logger.warning(f"[D] TaskStore 初始化失败，将使用文件轮询: {e}")
        # RAG 检索器
        self.rag = RAGRetriever()

    def _build_graph(self):
        """构建状态图"""
        from src.core.B import developer_node
        from src.core.E import evaluator_node
        from src.core.C import trainer_node

        workflow = StateGraph(SharedState)

        # 添加节点
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("develop", developer_node)
        workflow.add_node("evaluate", evaluator_node)
        workflow.add_node("train", trainer_node)
        workflow.add_node("human_review", self._human_review_node)

        # 定义条件边
        workflow.set_entry_point("analyze")

        # 分析 -> 开发
        workflow.add_conditional_edges(
            "analyze",
            self._route_after_analysis,
            {"develop": "develop", "human_review": "human_review"},
        )

        # 开发 -> 评估/人类审核
        workflow.add_conditional_edges(
            "develop",
            self._route_after_development,
            {"evaluate": "evaluate", "human_review": "human_review"},
        )

        # 评估 -> 训练/人类审核/重写
        workflow.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
            {"train": "train", "human_review": "human_review", "rewrite": "develop", "end": END},
        )

        # 训练 -> 人类审核
        workflow.add_edge("train", "human_review")

        # 人类审核 -> 继续/修改/结束
        workflow.add_conditional_edges(
            "human_review",
            self._route_after_human,
            {"continue": "develop", "train": "train", "end": END, "evaluate": "evaluate"},
        )

        return workflow.compile(checkpointer=self.checkpointer)

    def _thread_config(self, state: SharedState) -> dict:
        """生成 LangGraph checkpointer 配置"""
        thread_id = state.get("session_id") or state.get("task_id") or "default"
        return {"configurable": {"thread_id": thread_id}}

    def resume_session(self, session_id: str) -> SharedState:
        """从 checkpointer 断点续跑指定会话"""
        logger.info(f"[D] 尝试恢复会话: {session_id}")
        config = {"configurable": {"thread_id": session_id}}

        # 优先从已有 checkpoint 恢复
        checkpoint_tuple = self.checkpointer.get_tuple(config)
        if checkpoint_tuple is not None:
            state = dict(checkpoint_tuple.checkpoint.get("channel_values", {}))
            current_phase = state.get("current_phase", "init")
            logger.info(f"[D] 从 checkpoint 恢复会话: {session_id}, 阶段: {current_phase}")
            # 如果会话已结束，直接返回最终状态
            if current_phase in ("complete", "error", "failed"):
                logger.info(f"[D] 会话 {session_id} 已处于结束状态，无需续跑")
                return state
            result = self.workflow.invoke(state, config)
            return result

        # 兼容旧会话：从 SQLite sessions 表加载 state_json
        row = self.store.get_session(session_id)
        if not row or not row.get("state_json"):
            raise ValueError(f"找不到可恢复的会话: {session_id}")
        state = __import__("json").loads(row["state_json"])
        logger.info(f"[D] 从 sessions.db 恢复会话: {session_id}, 阶段: {state.get('current_phase')}")
        result = self.workflow.invoke(state, config)
        return result

    def _save_state(self, state: SharedState, phase: str) -> None:
        """保存状态、记录时间戳，并向 Redis 发布阶段事件"""
        now = datetime.now().isoformat()
        state["phase_timestamps"] = state.get("phase_timestamps", {})
        state["phase_timestamps"][phase] = now
        state["current_phase"] = phase

        try:
            metrics.observe_phase(phase)
        except Exception:
            pass

        try:
            self.store.save_session(state)
            state["db_saved"] = True
        except Exception as e:
            logger.warning(f"[D] 持久化失败: {e}")

        # 发布阶段事件到 Redis，供 SSE 实时推送
        task_id = state.get("task_id")
        if self._task_store and task_id:
            callbacks.publish_phase(self._task_store, task_id, phase, state)
            callbacks.check_cancel_signal(self._task_store, task_id)

    def run(self, requirements: str) -> SharedState:
        """运行完整流程"""
        initial_state = create_initial_state(requirements)
        if self.task_id:
            initial_state["task_id"] = self.task_id

        logger.info(f"[D] 启动编排器，session: {initial_state['session_id']}, task: {self.task_id}")

        start_time = datetime.now()

        config = self._thread_config(initial_state)

        try:
            result = self.workflow.invoke(initial_state, config)

            # 计算总耗时
            end_time = datetime.now()
            total_duration = int((end_time - start_time).total_seconds() * 1000)
            result["total_duration_ms"] = total_duration
            result["status"] = "complete"

            # 最终保存
            self._save_state(result, "complete")

            # 输出最终统计
            self._print_summary(result)

            logger.info(f"[D] 流程完成，状态: {result.get('status', 'unknown')}")
            return result

        except Exception as e:
            classified = classify_exception(e)
            error_type = type(classified).__name__
            logger.error(
                f"[D] 流程错误 [{error_type}]: {classified}",
                extra={
                    "error_type": error_type,
                    "recoverable": isinstance(classified, (TransientError, DegradableError)),
                },
            )
            initial_state["error"] = str(classified)
            initial_state["error_type"] = error_type
            initial_state["recoverable"] = isinstance(classified, (TransientError, DegradableError))

            if isinstance(classified, TransientError):
                initial_state["status"] = "retryable_error"
            elif isinstance(classified, DegradableError):
                initial_state["status"] = "degraded"
            elif isinstance(classified, FatalError):
                initial_state["status"] = "failed"
            else:
                initial_state["status"] = "error"

            self._save_state(initial_state, "error")
            return initial_state

    def _print_summary(self, state: SharedState) -> None:
        """打印流程总结"""
        print("\n" + "=" * 70)
        print("📋 流程总结")
        print("=" * 70)

        print(f"\n会话ID: {state.get('session_id', '')}")
        print(f"状态: {state.get('status', 'unknown')}")
        print(f"迭代次数: {state.get('iteration_count', 0)}")

        # 时间统计
        phases = state.get("phase_timestamps", {})
        if phases:
            print("\n⏱️ 各阶段时间:")
            for phase, ts in phases.items():
                print(f"  {phase}: {ts}")

        # LLM使用统计
        llm_usage = state.get("llm_usage", [])
        if llm_usage:
            total_tokens = sum(u.get("total_tokens", 0) for u in llm_usage)
            total_cost = sum(u.get("cost_usd", 0.0) for u in llm_usage)
            total_duration = sum(u.get("elapsed_ms", 0) for u in llm_usage)

            print(f"\n💰 LLM使用统计:")
            print(f"  总Tokens: {total_tokens:,}")
            print(f"  总成本: ${total_cost:.4f}")
            print(f"  LLM耗时: {total_duration}ms")

            # 按阶段分组
            by_phase = {}
            for u in llm_usage:
                p = u.get("phase", "unknown")
                by_phase[p] = by_phase.get(p, 0) + u.get("total_tokens", 0)

            print(f"  按阶段分布:")
            for phase, tokens in by_phase.items():
                print(f"    {phase}: {tokens:,} tokens")

        # 评估结果
        eval_result = state.get("evaluation_result")
        if eval_result:
            print(f"\n📊 评估结果:")
            print(f"  总体评分: {eval_result.get('overall_score', 0)}/100")
            print(f"  是否通过: {'✅' if eval_result.get('passed') else '❌'}")

            dims = eval_result.get("dimensions", {})
            if dims:
                print(f"  维度评分:")
                for dim_name, dim_data in dims.items():
                    if isinstance(dim_data, dict):
                        print(f"    {dim_name}: {dim_data.get('score', 0)}/100")

        # 代码产出
        artifacts = state.get("code_artifacts", [])
        if artifacts:
            print(f"\n📁 代码产出: {len(artifacts)} 个文件")
            for a in artifacts[:10]:
                print(f"  - {a.get('file_path', '')}")

        # 数据库存储
        if state.get("db_saved"):
            print(f"\n💾 已保存到 SQLite: data/sessions.db")

        print("=" * 70 + "\n")
