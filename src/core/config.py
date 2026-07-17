"""统一配置层（Pydantic Settings）

平台级配置的唯一入口，所有环境变量均在此定义默认值与映射关系。
业务代码通过 ``from src.core.config import settings`` 读取配置，
不再直接调用 ``os.getenv``。
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: str = "config/config.yaml") -> dict:
    """加载项目 YAML 配置。

    优先尝试 ``path``（相对于项目根目录），回退到 ``src/config.yaml``；
    文件不存在或解析失败时返回空字典，避免启动失败。
    """
    candidates = [_PROJECT_ROOT / path, _PROJECT_ROOT / "src" / "config.yaml"]
    for candidate in candidates:
        if candidate.exists():
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
    return {}


def _flatten_yaml(cfg: dict) -> dict:
    """从 YAML 中提取常用标量默认值。"""
    orchestrator_cfg = cfg.get("orchestrator", {})
    vector_store_cfg = cfg.get("vector_store", {})
    sft_cfg = cfg.get("sft", {})
    ml_eval_cfg = cfg.get("ml_eval", {})
    logging_cfg = cfg.get("logging", {})
    rag_cfg = cfg.get("rag", {})
    generation_cfg = cfg.get("generation", {})

    roles = sft_cfg.get("roles", ["D", "B", "E", "C"])
    if isinstance(roles, list):
        roles = ",".join(str(role) for role in roles)

    return {
        "max_iterations": orchestrator_cfg.get("max_iterations", 10),
        "vector_store_enabled": vector_store_cfg.get("enabled", True),
        "qdrant_url": vector_store_cfg.get("qdrant_url", "http://localhost:6333"),
        "qdrant_collection": vector_store_cfg.get("collection_name", "agent_memory"),
        "qdrant_local_path": vector_store_cfg.get("local_path", "data/qdrant_storage"),
        "qdrant_fallback_to_local": vector_store_cfg.get("fallback_to_local", True),
        "embedding_model": vector_store_cfg.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        "vector_size": vector_store_cfg.get("vector_size", 384),
        "sft_enabled": sft_cfg.get("enabled", True),
        "sft_min_score": sft_cfg.get("min_score", 70.0),
        "sft_min_samples": sft_cfg.get("min_samples", 50),
        "sft_roles": roles,
        "ml_eval_enabled": ml_eval_cfg.get("enabled", True),
        "ml_eval_mode": ml_eval_cfg.get("mode", "shadow"),
        "ml_eval_weight_ml": ml_eval_cfg.get("weight_ml", 0.3),
        "ml_eval_min_samples": ml_eval_cfg.get("min_samples", 20),
        "log_level": logging_cfg.get("level", "INFO"),
        "log_file": logging_cfg.get("file", "logs/agent_system.log"),
        "sft_auto_train": sft_cfg.get("auto_train", False),
        "sft_mlx_timeout": sft_cfg.get("mlx_timeout", 3600),
        "sft_mlx_max_retries": sft_cfg.get("mlx_max_retries", 2),
        "mcp_native_function_calling": cfg.get("mcp", {}).get("native_function_calling", False),
        "mcp_max_tool_rounds": cfg.get("mcp", {}).get("max_tool_rounds", 5),
        "projects_dir": cfg.get("projects", {}).get("dir", "projects"),
        "rag_rerank_model": rag_cfg.get("rerank_model"),
        "split_threshold_tokens": generation_cfg.get("split_threshold_tokens", 3000),
        "chunk_max_tokens": generation_cfg.get("chunk_max_tokens", 3072),
        "max_completion_tokens_simple": generation_cfg.get(
            "max_completion_tokens_simple", 4096
        ),
        "max_completion_tokens_normal": generation_cfg.get(
            "max_completion_tokens_normal", 4096
        ),
        "max_completion_tokens_complex": generation_cfg.get(
            "max_completion_tokens_complex", 8192
        ),
    }


_YAML_CFG = _load_yaml()
_FLAT_CFG = _flatten_yaml(_YAML_CFG)


class Settings(BaseSettings):
    """Pydantic Settings 配置类。

    环境变量名与字段通过 ``Field(alias=...)`` 显式映射，保持与历史配置一致。
    ``populate_by_name=True`` 允许通过字段名初始化，便于构造时显式覆盖。
    标量默认值优先从 ``config/config.yaml`` 读取，仍可由环境变量覆盖。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    # 数据层与队列
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    database_url: str = Field(default="sqlite:///data/platform.db", alias="DATABASE_URL")

    # Kimi Code API
    kimi_api_key: str = Field(default="", alias="KIMI_API_KEY")
    kimi_base_url: str = Field(
        default="https://api.kimi.com/coding/v1", alias="KIMI_BASE_URL"
    )
    kimi_model: str = Field(default="kimi-k2p5", alias="KIMI_MODEL")
    kimi_api_format: Optional[str] = Field(default=None, alias="KIMI_API_FORMAT")
    auto_approve: bool = Field(default=False, alias="AUTO_APPROVE")

    # 编排器
    max_iterations: int = Field(
        default=_FLAT_CFG["max_iterations"], alias="MAX_ITERATIONS"
    )

    # 代码生成（B 阶段）
    split_threshold_tokens: int = Field(
        default=_FLAT_CFG["split_threshold_tokens"], alias="SPLIT_THRESHOLD_TOKENS"
    )
    chunk_max_tokens: int = Field(
        default=_FLAT_CFG["chunk_max_tokens"], alias="CHUNK_MAX_TOKENS"
    )
    max_completion_tokens_simple: int = Field(
        default=_FLAT_CFG["max_completion_tokens_simple"],
        alias="MAX_COMPLETION_TOKENS_SIMPLE",
    )
    max_completion_tokens_normal: int = Field(
        default=_FLAT_CFG["max_completion_tokens_normal"],
        alias="MAX_COMPLETION_TOKENS_NORMAL",
    )
    max_completion_tokens_complex: int = Field(
        default=_FLAT_CFG["max_completion_tokens_complex"],
        alias="MAX_COMPLETION_TOKENS_COMPLEX",
    )

    # 向量数据库与 RAG
    vector_store_enabled: bool = Field(
        default=_FLAT_CFG["vector_store_enabled"], alias="VECTOR_STORE_ENABLED"
    )
    qdrant_url: str = Field(default=_FLAT_CFG["qdrant_url"], alias="QDRANT_URL")
    qdrant_collection: str = Field(
        default=_FLAT_CFG["qdrant_collection"], alias="QDRANT_COLLECTION"
    )
    embedding_model: str = Field(
        default=_FLAT_CFG["embedding_model"], alias="EMBEDDING_MODEL"
    )
    vector_size: Optional[int] = Field(
        default=_FLAT_CFG["vector_size"], alias="VECTOR_SIZE"
    )
    qdrant_local_path: str = Field(
        default=_FLAT_CFG["qdrant_local_path"], alias="QDRANT_LOCAL_PATH"
    )
    qdrant_fallback_to_local: bool = Field(
        default=_FLAT_CFG["qdrant_fallback_to_local"], alias="QDRANT_FALLBACK_TO_LOCAL"
    )

    # MCP
    mcp_enabled: bool = Field(default=True, alias="MCP_ENABLED")
    mcp_shell_allow_local: bool = Field(default=False, alias="MCP_SHELL_ALLOW_LOCAL")

    # SFT 数据飞轮
    sft_enabled: bool = Field(default=_FLAT_CFG["sft_enabled"], alias="SFT_ENABLED")
    sft_roles: str = Field(default=_FLAT_CFG["sft_roles"], alias="SFT_ROLES")
    sft_min_score: float = Field(
        default=_FLAT_CFG["sft_min_score"], alias="SFT_MIN_SCORE"
    )
    sft_min_samples: int = Field(
        default=_FLAT_CFG["sft_min_samples"], alias="SFT_MIN_SAMPLES"
    )

    # ML 自动化评估
    ml_eval_enabled: bool = Field(
        default=_FLAT_CFG["ml_eval_enabled"], alias="ML_EVAL_ENABLED"
    )
    ml_eval_mode: str = Field(
        default=_FLAT_CFG["ml_eval_mode"], alias="ML_EVAL_MODE"
    )
    ml_eval_weight_ml: float = Field(
        default=_FLAT_CFG["ml_eval_weight_ml"], alias="ML_EVAL_WEIGHT_ML"
    )
    ml_eval_min_samples: int = Field(
        default=_FLAT_CFG["ml_eval_min_samples"], alias="ML_EVAL_MIN_SAMPLES"
    )
    ml_eval_auto_stage: str = Field(default="shadow", alias="ML_EVAL_AUTO_STAGE")

    # 日志（P1.3 使用）
    log_level: str = Field(default=_FLAT_CFG["log_level"], alias="LOG_LEVEL")
    log_file: Optional[str] = Field(default=_FLAT_CFG["log_file"], alias="LOG_FILE")
    json_log_format: bool = Field(default=True, alias="JSON_LOG_FORMAT")

    # 指标（P2.3 使用）
    metrics_port: int = Field(default=9100, alias="METRICS_PORT")

    # HITL 超时配置（Q3）
    hitl_timeout_seconds: int = Field(default=1800, alias="HITL_TIMEOUT_SECONDS")
    hitl_timeout_action: str = Field(default="approve", alias="HITL_TIMEOUT_ACTION")

    # Phase 2: 评估阈值与质量门
    eval_pass_threshold: float = Field(default=80.0, alias="EVAL_PASS_THRESHOLD")
    eval_simple_project_threshold: float = Field(
        default=50.0, alias="EVAL_SIMPLE_PROJECT_THRESHOLD"
    )
    ml_eval_disagreement_threshold: float = Field(
        default=15.0, alias="ML_EVAL_DISAGREEMENT_THRESHOLD"
    )
    ml_eval_severe_disagreement_threshold: float = Field(
        default=30.0, alias="ML_EVAL_SEVERE_DISAGREEMENT_THRESHOLD"
    )
    sft_active_learning_enabled: bool = Field(
        default=False, alias="SFT_ACTIVE_LEARNING_ENABLED"
    )
    sft_active_learning_subset_size: int = Field(
        default=100, alias="SFT_ACTIVE_LEARNING_SUBSET_SIZE"
    )
    sft_per_role_min_samples: int = Field(
        default=50, alias="SFT_PER_ROLE_MIN_SAMPLES"
    )
    sft_max_duplicate_ratio: float = Field(
        default=0.3, alias="SFT_MAX_DUPLICATE_RATIO"
    )
    sft_min_avg_score: float = Field(default=75.0, alias="SFT_MIN_AVG_SCORE")
    sft_auto_train: bool = Field(
        default=_FLAT_CFG["sft_auto_train"], alias="SFT_AUTO_TRAIN"
    )
    sft_mlx_timeout: int = Field(
        default=_FLAT_CFG["sft_mlx_timeout"], alias="SFT_MLX_TIMEOUT"
    )
    sft_mlx_max_retries: int = Field(
        default=_FLAT_CFG["sft_mlx_max_retries"], alias="SFT_MLX_MAX_RETRIES"
    )

    # MCP 原生 function calling
    mcp_native_function_calling: bool = Field(
        default=_FLAT_CFG["mcp_native_function_calling"],
        alias="MCP_NATIVE_FUNCTION_CALLING",
    )
    mcp_max_tool_rounds: int = Field(
        default=_FLAT_CFG["mcp_max_tool_rounds"], alias="MCP_MAX_TOOL_ROUNDS"
    )

    # Phase 5: RAG 数据种子与检索
    projects_dir: str = Field(
        default=_FLAT_CFG["projects_dir"], alias="PROJECTS_DIR"
    )
    rag_rerank_model: Optional[str] = Field(
        default=_FLAT_CFG["rag_rerank_model"], alias="RAG_RERANK_MODEL"
    )

    # 结构化 YAML 配置段（可选，便于业务代码读取完整配置块）
    orchestrator_config: Optional[dict] = Field(
        default_factory=lambda: _YAML_CFG.get("orchestrator", {}).copy()
    )
    agents_config: Optional[dict] = Field(
        default_factory=lambda: _YAML_CFG.get("agents", {}).copy()
    )
    evaluation_dimensions: Optional[dict] = Field(
        default_factory=lambda: _YAML_CFG.get("evaluation", {})
        .get("dimensions", {})
        .copy()
    )
    mcp_tools_config: Optional[dict] = Field(
        default_factory=lambda: _YAML_CFG.get("mcp", {}).get("tools", {}).copy()
    )
    gateway_config: Optional[dict] = Field(
        default_factory=lambda: _YAML_CFG.get("gateway", {}).copy()
    )
    logging_config: Optional[dict] = Field(
        default_factory=lambda: _YAML_CFG.get("logging", {}).copy()
    )

    @field_validator("hitl_timeout_action")
    @classmethod
    def _validate_hitl_timeout_action(cls, v: str) -> str:
        allowed = {"approve", "decline", "abort"}
        if v not in allowed:
            raise ValueError(f"HITL_TIMEOUT_ACTION must be one of {allowed}, got {v!r}")
        return v


class _SettingsProxy:
    """配置代理，每次访问属性时重新实例化 ``Settings``。

    这样做可以在运行期或测试的 monkeypatch 修改环境变量后仍读取到最新值，
    同时保持 ``settings.xxx`` 的调用方式不变。
    """

    def __getattr__(self, name: str):
        return getattr(Settings(), name)


# 业务代码统一导入此单例
settings = _SettingsProxy()
