"""persistence.py - SQLite持久化存储

P2核心功能：
- 会话持久化（SharedState -> SQLite）
- 评估历史查询
- 成本/时间追踪统计
- 项目产出管理
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.core.logger import logger


class SessionStore:
    """SQLite会话存储"""
    
    def __init__(self, db_path: str = "data/sessions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """初始化数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            # 会话表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    requirements TEXT,
                    status TEXT,
                    current_phase TEXT,
                    project_name TEXT,
                    overall_score REAL,
                    passed INTEGER,
                    iteration_count INTEGER,
                    total_tokens INTEGER,
                    total_cost REAL,
                    total_duration_ms INTEGER,
                    state_json TEXT
                )
            """)
            
            # LLM使用记录
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_usd REAL,
                    elapsed_ms INTEGER,
                    created_at TEXT
                )
            """)
            
            # 代码产出
            conn.execute("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    iteration INTEGER,
                    file_path TEXT NOT NULL,
                    content TEXT,
                    language TEXT,
                    source TEXT,
                    created_at TEXT
                )
            """)
            
            # 评估结果历史
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    iteration INTEGER,
                    overall_score REAL,
                    code_quality REAL,
                    functionality REAL,
                    performance REAL,
                    documentation REAL,
                    passed INTEGER,
                    issues_json TEXT,
                    recommendations_json TEXT,
                    created_at TEXT
                )
            """)
            
            # 测试运行结果
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    iteration INTEGER,
                    file TEXT,
                    passed INTEGER,
                    duration_ms INTEGER,
                    output TEXT,
                    error TEXT,
                    sandbox TEXT,
                    created_at TEXT
                )
            """)
            
            conn.commit()
            logger.info(f"[DB] 数据库初始化完成: {self.db_path}")
    
    def save_session(self, state: dict) -> None:
        """保存会话状态"""
        session_id = state.get("session_id", "")
        if not session_id:
            logger.warning("[DB] 会话ID为空，跳过保存")
            return
        
        now = datetime.now().isoformat()
        
        # 计算汇总统计
        llm_usage = state.get("llm_usage", [])
        total_tokens = sum(u.get("total_tokens", 0) for u in llm_usage)
        total_cost = sum(u.get("cost_usd", 0.0) for u in llm_usage)
        total_duration = sum(u.get("elapsed_ms", 0) for u in llm_usage)
        
        eval_result = state.get("evaluation_result", {})
        
        with sqlite3.connect(self.db_path) as conn:
            # 先清除该会话的历史子记录，避免重复写入
            # 会话主表使用 INSERT OR REPLACE，不需要先删除
            for table in ("llm_usage", "artifacts", "evaluations", "test_runs"):
                conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))

            # 保存/更新会话
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, created_at, updated_at, requirements, status, current_phase,
                 project_name, overall_score, passed, iteration_count,
                 total_tokens, total_cost, total_duration_ms, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                state.get("created_at", now),
                now,
                state.get("requirements", ""),
                state.get("status", ""),
                state.get("current_phase", ""),
                state.get("project", {}).get("name", ""),
                eval_result.get("overall_score", 0) if eval_result else 0,
                1 if (eval_result.get("passed", False) if eval_result else False) else 0,
                state.get("iteration_count", 0),
                total_tokens,
                total_cost,
                total_duration,
                json.dumps(state, default=str)
            ))
            
            # 保存LLM使用记录
            for usage in llm_usage:
                conn.execute("""
                    INSERT INTO llm_usage
                    (session_id, phase, input_tokens, output_tokens, total_tokens, cost_usd, elapsed_ms, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    usage.get("phase", "unknown"),
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("total_tokens", 0),
                    usage.get("cost_usd", 0.0),
                    usage.get("elapsed_ms", 0),
                    now
                ))
            
            # 保存代码产出（仅最新迭代的）
            iteration = state.get("iteration_count", 0)
            for artifact in state.get("code_artifacts", []):
                conn.execute("""
                    INSERT INTO artifacts
                    (session_id, iteration, file_path, content, language, source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    iteration,
                    artifact.get("file_path", ""),
                    artifact.get("content", ""),
                    artifact.get("language", "text"),
                    artifact.get("source", "unknown"),
                    now
                ))
            
            # 保存评估结果
            if eval_result:
                dims = eval_result.get("dimensions", {})
                conn.execute("""
                    INSERT INTO evaluations
                    (session_id, iteration, overall_score, code_quality, functionality, performance, documentation, passed, issues_json, recommendations_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    iteration,
                    eval_result.get("overall_score", 0),
                    dims.get("code_quality", {}).get("score", 0) if isinstance(dims.get("code_quality"), dict) else dims.get("code_quality", 0),
                    dims.get("functionality", {}).get("score", 0) if isinstance(dims.get("functionality"), dict) else dims.get("functionality", 0),
                    dims.get("performance", {}).get("score", 0) if isinstance(dims.get("performance"), dict) else dims.get("performance", 0),
                    dims.get("documentation", {}).get("score", 0) if isinstance(dims.get("documentation"), dict) else dims.get("documentation", 0),
                    1 if eval_result.get("passed", False) else 0,
                    json.dumps(eval_result.get("issues", [])),
                    json.dumps(eval_result.get("recommendations", [])),
                    now
                ))
            
            # 保存测试结果
            for test in state.get("test_results", [])[-5:]:  # 最近5条
                conn.execute("""
                    INSERT INTO test_runs
                    (session_id, iteration, file, passed, duration_ms, output, error, sandbox, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    iteration,
                    test.get("file", ""),
                    1 if test.get("passed", False) else 0,
                    test.get("duration_ms", 0),
                    test.get("output", ""),
                    test.get("error", ""),
                    test.get("sandbox", "unknown"),
                    now
                ))
            
            conn.commit()
        
        logger.info(f"[DB] 会话已保存: {session_id}")
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """获取会话状态"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
            
            if row:
                return dict(row)
            return None
    
    def list_sessions(self, limit: int = 50) -> List[dict]:
        """列出会话历史"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    
    def get_cost_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """获取成本汇总"""
        with sqlite3.connect(self.db_path) as conn:
            if session_id:
                row = conn.execute("""
                    SELECT 
                        SUM(total_tokens) as total_tokens,
                        SUM(total_cost) as total_cost,
                        SUM(total_duration_ms) as total_duration_ms,
                        COUNT(*) as session_count
                    FROM sessions WHERE session_id = ?
                """, (session_id,)).fetchone()
            else:
                row = conn.execute("""
                    SELECT 
                        SUM(total_tokens) as total_tokens,
                        SUM(total_cost) as total_cost,
                        SUM(total_duration_ms) as total_duration_ms,
                        COUNT(*) as session_count
                    FROM sessions
                """).fetchone()
            
            return {
                "total_tokens": row[0] or 0,
                "total_cost_usd": row[1] or 0.0,
                "total_duration_ms": row[2] or 0,
                "session_count": row[3] or 0
            }
    
    def get_performance_stats(self, session_id: str) -> Dict[str, Any]:
        """获取性能统计"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # LLM各阶段统计
            llm_rows = conn.execute("""
                SELECT phase, 
                       SUM(total_tokens) as tokens,
                       SUM(cost_usd) as cost,
                       SUM(elapsed_ms) as duration,
                       COUNT(*) as calls
                FROM llm_usage WHERE session_id = ?
                GROUP BY phase
            """, (session_id,)).fetchall()
            
            # 测试统计
            test_rows = conn.execute("""
                SELECT passed, COUNT(*) as count, AVG(duration_ms) as avg_duration
                FROM test_runs WHERE session_id = ?
                GROUP BY passed
            """, (session_id,)).fetchall()
            
            # 评估趋势
            eval_rows = conn.execute("""
                SELECT iteration, overall_score, passed
                FROM evaluations WHERE session_id = ?
                ORDER BY iteration
            """, (session_id,)).fetchall()
            
            return {
                "llm_by_phase": [dict(r) for r in llm_rows],
                "test_summary": [dict(r) for r in test_rows],
                "evaluation_trend": [dict(r) for r in eval_rows]
            }


# 全局存储实例
_db: Optional[SessionStore] = None

def get_session_store() -> SessionStore:
    """获取存储实例（单例）"""
    global _db
    if _db is None:
        _db = SessionStore()
    return _db
