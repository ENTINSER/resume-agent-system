"""共享状态定义 - 4智能体平台核心"""

from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime
import uuid

from src.core.config import settings


class SharedState(TypedDict, total=False):
    """4智能体共享状态 - 中央数据库"""
    
    # 会话标识
    session_id: str
    created_at: str
    
    # 项目信息
    project: Optional[dict]
    
    # D - Orchestrator 状态
    current_phase: str
    milestone_index: int
    iteration_count: int
    max_iterations: int
    
    # B - Developer 状态
    requirements: str
    code_artifacts: List[dict]
    test_results: List[dict]
    repo_url: Optional[str]
    commit_history: List[str]
    
    # E - Evaluator 状态
    evaluation_result: Optional[dict]
    evaluation_history: List[dict]
    
    # C - Trainer 状态
    training_jobs: List[dict]
    model_path: Optional[str]
    dataset_stats: Optional[dict]
    
    # 人类决策
    human_approved: Optional[bool]
    human_feedback: str
    human_review_required: bool
    human_review_reason: str

    # B 阶段修复追踪
    repair_history: Dict[str, int]
    import_errors: List[str]

    # 骨架生成
    skeleton_complete: bool
    skeleton_artifacts: List[dict]
    
    # LLM 使用追踪
    llm_usage: List[dict]
    
    # 时间追踪
    phase_timestamps: Dict[str, str]
    total_duration_ms: int
    
    # P0修复：格式失败计数
    format_failures: int

    # 持久化
    db_saved: bool

    # MCP 工具调用
    available_tools: List[str]
    tool_calls: List[dict]
    tool_results: List[dict]

    # ML 自动化评估
    static_metrics: Optional[dict]
    ml_features: Optional[dict]
    ml_score: Optional[float]
    ml_model_id: Optional[str]
    ml_mode: Optional[str]
    evaluation_source: Optional[str]
    dataset_saved: bool

    # SFT 数据飞轮
    sft_samples: List[dict]
    sft_dataset_saved: bool
    sft_training_jobs: List[dict]

    # 通用状态字段
    status: str
    error: Optional[str]
    logs: List[dict]


def create_initial_state(requirements: str = "") -> SharedState:
    """创建初始状态"""
    return {
        "session_id": str(uuid.uuid4()),
        "created_at": datetime.now().isoformat(),
        "project": None,
        "current_phase": "init",
        "milestone_index": 0,
        "iteration_count": 0,
        "max_iterations": settings.max_iterations,
        "requirements": requirements,
        "code_artifacts": [],
        "test_results": [],
        "repo_url": None,
        "commit_history": [],
        "evaluation_result": None,
        "evaluation_history": [],
        "training_jobs": [],
        "model_path": None,
        "dataset_stats": None,
        "human_approved": None,
        "human_feedback": "",
        "human_review_required": False,
        "human_review_reason": "",
        "repair_history": {},
        "import_errors": [],
        "skeleton_complete": False,
        "skeleton_artifacts": [],
        "llm_usage": [],
        "phase_timestamps": {},
        "total_duration_ms": 0,
        "format_failures": 0,
        "db_saved": False,
        "available_tools": [],
        "tool_calls": [],
        "tool_results": [],
        "static_metrics": None,
        "ml_features": None,
        "ml_score": None,
        "ml_model_id": None,
        "ml_mode": None,
        "evaluation_source": None,
        "dataset_saved": False,
        "sft_samples": [],
        "sft_dataset_saved": False,
        "sft_training_jobs": [],
        "status": "setup",
        "error": None,
        "logs": [],
    }
