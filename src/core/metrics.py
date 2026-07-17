"""Prometheus 指标暴露

Worker 启动时可通过 `start_metrics_server()` 开启 /metrics HTTP 端点，
供 Prometheus 抓取任务级指标。
"""

from typing import Optional

from prometheus_client import Counter, Histogram, start_http_server

from src.core.config import settings

# 任务维度
tasks_total = Counter(
    "agent_tasks_total",
    "已处理任务总数",
    ["status"],
)

task_duration_seconds = Histogram(
    "agent_task_duration_seconds",
    "任务执行耗时",
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800],
)

# LLM 维度
llm_tokens_total = Counter(
    "agent_llm_tokens_total",
    "LLM 累计 token 数",
    ["phase"],
)

llm_cost_dollars_total = Counter(
    "agent_llm_cost_dollars_total",
    "LLM 累计成本（美元）",
)

# 阶段维度
phase_transitions_total = Counter(
    "agent_phase_transitions_total",
    "阶段切换次数",
    ["phase"],
)

# ML/LLM 评估分歧维度
ml_eval_disagreements_total = Counter(
    "ml_eval_disagreements_total",
    "LLM/ML 评估分歧次数",
    ["level"],
)

# B 阶段截断与修复维度
b_truncations_total = Counter(
    "agent_b_truncations_total",
    "B 阶段文件截断次数",
    ["status"],  # truncated / continued / incomplete
)

b_repair_attempts_total = Counter(
    "agent_b_repair_attempts_total",
    "B 阶段自修复尝试次数",
    ["file", "issue_type"],
)

b_hitl_escalations_total = Counter(
    "agent_b_hitl_escalations_total",
    "B 阶段升级到 HITL 次数",
    ["reason"],
)

# C 阶段样本候选维度
c_sample_candidates_total = Counter(
    "agent_c_sample_candidates_total",
    "进入 ML/SFT 飞轮的样本候选数",
    ["passed"],
)

_metrics_server_started = False


def start_metrics_server(port: Optional[int] = None) -> None:
    """启动 Prometheus HTTP 指标服务，幂等"""
    global _metrics_server_started
    if _metrics_server_started:
        return

    port = port or getattr(settings, "metrics_port", 9100)
    if not port:
        return

    try:
        start_http_server(int(port))
        _metrics_server_started = True
    except Exception as exc:
        from src.core.logger import logger
        logger.warning(f"[Metrics] 启动指标服务失败: {exc}")


def observe_task(status: str, duration_seconds: float) -> None:
    """记录任务完成指标"""
    tasks_total.labels(status=status).inc()
    task_duration_seconds.observe(duration_seconds)


def observe_llm(phase: str, tokens: int, cost: float) -> None:
    """记录 LLM 调用指标"""
    llm_tokens_total.labels(phase=phase).inc(tokens)
    llm_cost_dollars_total.inc(cost)


def observe_phase(phase: str) -> None:
    """记录阶段切换指标"""
    phase_transitions_total.labels(phase=phase).inc()


def observe_ml_disagreement(level: str) -> None:
    """记录 LLM/ML 评估分歧指标"""
    ml_eval_disagreements_total.labels(level=level).inc()


def observe_b_truncation(status: str) -> None:
    """记录 B 阶段文件截断指标"""
    b_truncations_total.labels(status=status).inc()


def observe_b_repair_attempt(file: str, issue_type: str) -> None:
    """记录 B 阶段自修复尝试指标"""
    b_repair_attempts_total.labels(file=file, issue_type=issue_type).inc()


def observe_b_hitl(reason: str) -> None:
    """记录 B 阶段升级到 HITL 指标"""
    b_hitl_escalations_total.labels(reason=reason).inc()


def observe_c_sample_candidate(passed: bool) -> None:
    """记录 C 阶段样本候选指标"""
    c_sample_candidates_total.labels(passed=str(passed)).inc()
