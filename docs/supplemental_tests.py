#!/usr/bin/env python3
"""补充测试：稳定性回归、token/耗时基准、大文件压力、重试上限。"""

import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List

BASE_URL = "http://localhost:8080/api/v1/tasks"
PROJECTS_ROOT = Path("/Users/mingrun/resume-agent-system/projects")
LOG_PATH = Path("/Users/mingrun/resume-agent-system/logs/celery.log")
RESULT_PATH = Path("/Users/mingrun/resume-agent-system/docs/supplemental_test_results.json")

SIMPLE_CLI_REQ = "实现一个命令行 Todo 工具：支持 add、list、done、delete 四个命令，数据用 JSON 文件本地持久化，要求 pytest 测试通过。"
LANGGRAPH_REQ = "实现一个 LangGraph 多智能体文章写作系统：包含 researcher、writer、reviewer 三个角色，能协作完成一篇文章；reviewer 从准确性、完整性、可读性、结构四个维度评分（满分100，通过阈值80），最多修改3次；要求代码可运行，pytest 测试通过，README 完整。"
STRESS_REQ = "实现一个 Python 模块 src/string_utils.py，包含至少 25 个独立的字符串处理工具函数，每个函数不少于 15 行代码，并配套 pytest 测试。"


def submit_task(requirements: str, project_name: str) -> str:
    """提交任务并返回 task_id。"""
    full_req = f"在 projects/{project_name} 目录下，{requirements}"
    payload = json.dumps({"requirements": full_req})
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", BASE_URL, "-H", "Content-Type: application/json", "-d", payload],
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    return data["task_id"]


def wait_for_task(task_id: str, timeout: int = 3600) -> Dict:
    """轮询任务直到完成或超时。"""
    start = time.time()
    while time.time() - start < timeout:
        result = subprocess.run(
            ["curl", "-s", f"{BASE_URL}/{task_id}"],
            capture_output=True,
            text=True,
        )
        try:
            data = json.loads(result.stdout)
        except Exception:
            time.sleep(5)
            continue
        status = data.get("status", "unknown")
        if status in ("completed", "failed", "cancelled"):
            return data
        time.sleep(10)
    return {"status": "timeout", "task_id": task_id}


def get_log_lines_since(start_line: int) -> str:
    """读取 celery.log 中从 start_line 开始的新行。"""
    if not LOG_PATH.exists():
        return ""
    lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(lines[start_line:])


def analyze_run(task_data: Dict, project_dir: Path, log_segment: str) -> Dict:
    """从任务数据、项目目录和日志片段中提取指标。"""
    result = task_data.get("result", {})
    eval_res = result.get("evaluation_result", {}) or {}
    metrics = {
        "task_id": task_data.get("task_id"),
        "status": task_data.get("status"),
        "score": eval_res.get("overall_score"),
        "passed": eval_res.get("passed"),
        "iteration_count": result.get("iteration_count"),
        "duration_ms": result.get("total_duration_ms"),
        "duration_min": round((result.get("total_duration_ms") or 0) / 60000, 2),
        "llm_calls": len(re.findall(r"\[LLM\] 生成完成", log_segment)),
        "timeouts": len(re.findall(r"Read timed out", log_segment)),
        "chunk_retries": len(re.findall(r"校验失败，重试", log_segment)),
        "fallbacks": len(re.findall(r"回退到单文件生成", log_segment)),
        "stub_rejections": len(re.findall(r"存在空方法体", log_segment)),
    }

    # token 消耗估算
    prompt_tokens = 0
    completion_tokens = 0
    for line in log_segment.splitlines():
        m = re.search(r"prompt=(\d+).*completion=(\d+)", line)
        if m:
            prompt_tokens += int(m.group(1))
            completion_tokens += int(m.group(2))
    metrics["prompt_tokens"] = prompt_tokens
    metrics["completion_tokens"] = completion_tokens
    metrics["total_tokens"] = prompt_tokens + completion_tokens

    # 文件规模
    if project_dir.exists():
        max_lines = 0
        total_lines = 0
        for f in project_dir.rglob("*.py"):
            try:
                lc = len(f.read_text(encoding="utf-8").splitlines())
                max_lines = max(max_lines, lc)
                total_lines += lc
            except Exception:
                pass
        metrics["max_file_lines"] = max_lines
        metrics["total_python_lines"] = total_lines
    else:
        metrics["max_file_lines"] = 0
        metrics["total_python_lines"] = 0

    return metrics


def run_series(name: str, requirements: str, prefix: str, count: int, timeout: int) -> List[Dict]:
    """连续运行同一任务多次，收集指标。"""
    runs = []
    for i in range(count):
        project_name = f"{prefix}-run{i+1}"
        print(f"[{datetime.now()}] 启动 {name} run {i+1}/{count}: {project_name}")
        start_line = len(LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()) if LOG_PATH.exists() else 0
        task_id = submit_task(requirements, project_name)
        print(f"  task_id={task_id}")
        task_data = wait_for_task(task_id, timeout=timeout)
        time.sleep(2)
        log_segment = get_log_lines_since(start_line)
        project_dir = PROJECTS_ROOT / project_name
        metrics = analyze_run(task_data, project_dir, log_segment)
        metrics["run_index"] = i + 1
        metrics["task_name"] = name
        runs.append(metrics)
        print(f"  完成: score={metrics['score']}, duration={metrics['duration_min']}min, tokens={metrics['total_tokens']}")
    return runs


def summarize(runs: List[Dict]) -> Dict:
    """计算均值、方差等统计量。"""
    scores = [r["score"] for r in runs if r["score"] is not None]
    durations = [r["duration_min"] for r in runs if r["duration_min"] is not None]
    tokens = [r["total_tokens"] for r in runs]
    return {
        "count": len(runs),
        "score_mean": round(sum(scores) / len(scores), 2) if scores else None,
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_variance": round(sum((x - sum(scores)/len(scores))**2 for x in scores)/len(scores), 2) if scores else None,
        "duration_mean_min": round(sum(durations) / len(durations), 2) if durations else None,
        "token_mean": round(sum(tokens) / len(tokens)) if tokens else None,
        "timeout_total": sum(r["timeouts"] for r in runs),
        "fallback_total": sum(r["fallbacks"] for r in runs),
        "stub_rejection_total": sum(r["stub_rejections"] for r in runs),
    }


def main():
    # 清理 Redis，确保队列干净
    subprocess.run(["redis-cli", "FLUSHALL"], capture_output=True)
    time.sleep(1)

    results = {
        "simple_cli": run_series("Simple CLI", SIMPLE_CLI_REQ, "test-simple-cli", 3, timeout=1800),
        "langgraph": run_series("LangGraph", LANGGRAPH_REQ, "01-langgraph-multi-agent", 3, timeout=3600),
        "stress_large_file": run_series("Large File Stress", STRESS_REQ, "stress-large-file", 1, timeout=1800),
    }

    results["simple_cli_summary"] = summarize(results["simple_cli"])
    results["langgraph_summary"] = summarize(results["langgraph"])
    results["stress_large_file_summary"] = summarize(results["stress_large_file"])

    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n测试结果已保存到: {RESULT_PATH}")

    # 打印摘要
    for key in ["simple_cli_summary", "langgraph_summary", "stress_large_file_summary"]:
        print(f"\n{key}:")
        print(json.dumps(results[key], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
