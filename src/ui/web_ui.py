"""web_ui.py - FastAPI 可视化界面

方案D：FastAPI + 文件轮询
- 前端每3秒轮询检测 human review 状态
- 平台运行到 D 的 human_review 节点时写入 data/pending_review.json
- 用户点击按钮后写入 data/human_response.json
- 平台读取 response 后继续
"""

import os
import json
import threading
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from src.core.logger import logger
from src.core.persistence import get_session_store


PENDING_REVIEW_FILE = "data/pending_review.json"
HUMAN_RESPONSE_FILE = "data/human_response.json"

app = FastAPI(title="4智能体协作平台 - 控制面板")

# 静态文件和模板
app.mount("/static", StaticFiles(directory="src/ui/static"), name="static")
templates = Jinja2Templates(directory="src/ui/templates")

# 全局任务状态
_task_status = {"running": False, "session_id": None, "message": ""}


def run_task_in_background(requirements: str):
    """后台运行编排器任务"""
    global _task_status
    _task_status["running"] = True
    _task_status["message"] = "任务启动中..."
    
    try:
        from src.core.D import Orchestrator
        from src.core.B import save_project
        from src.core.E import generate_html_report
        
        orch = Orchestrator(ui_mode=True)
        result = orch.run(requirements)
        
        _task_status["session_id"] = result.get("session_id", "")
        _task_status["message"] = f"任务完成: {result.get('status', 'unknown')}"
        
        # 保存项目
        code_artifacts = result.get("code_artifacts", [])
        if code_artifacts:
            save_project(result, "projects")
        
    except Exception as e:
        logger.error(f"[UI] 任务失败: {e}")
        _task_status["message"] = f"任务失败: {e}"
    finally:
        _task_status["running"] = False
        # 清理 pending 文件
        for f in [PENDING_REVIEW_FILE, HUMAN_RESPONSE_FILE]:
            if os.path.exists(f):
                os.remove(f)
                logger.info(f"[UI] 清理文件: {f}")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页面"""
    store = get_session_store()
    sessions = store.list_sessions(limit=20)
    
    # 检查是否有待审核任务
    pending = None
    if os.path.exists(PENDING_REVIEW_FILE):
        try:
            with open(PENDING_REVIEW_FILE, 'r') as f:
                pending = json.load(f)
        except Exception:
            pass
    
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "sessions": sessions,
            "pending": pending,
            "task_running": _task_status["running"],
            "task_message": _task_status["message"],
        }
    )


@app.get("/api/status")
async def api_status():
    """API状态检查"""
    pending = None
    if os.path.exists(PENDING_REVIEW_FILE):
        try:
            with open(PENDING_REVIEW_FILE, 'r') as f:
                pending = json.load(f)
        except Exception:
            pass
    
    return JSONResponse({
        "has_pending": pending is not None,
        "pending": pending,
        "task_running": _task_status["running"],
        "task_message": _task_status["message"],
        "timestamp": datetime.now().isoformat(),
    })


@app.post("/api/start_task")
async def api_start_task(requirements: str = Form(...)):
    """启动新任务"""
    if _task_status["running"]:
        return JSONResponse({"error": "已有任务在运行中"}, status_code=400)
    
    if not requirements.strip():
        return JSONResponse({"error": "需求不能为空"}, status_code=400)
    
    # 在后台线程启动任务
    thread = threading.Thread(target=run_task_in_background, args=(requirements,))
    thread.daemon = True
    thread.start()
    
    logger.info(f"[UI] 启动新任务: {requirements[:50]}...")
    
    return JSONResponse({"status": "ok", "message": "任务已启动"})


@app.post("/api/respond")
async def api_respond(feedback: str = Form(...)):
    """用户响应接口"""
    if feedback not in ("approve", "rewrite", "evaluate", "train"):
        return JSONResponse({"error": "无效的反馈"}, status_code=400)
    
    response_data = {
        "feedback": feedback,
        "timestamp": datetime.now().isoformat(),
    }
    
    os.makedirs("data", exist_ok=True)
    with open(HUMAN_RESPONSE_FILE, 'w') as f:
        json.dump(response_data, f, ensure_ascii=False)
    
    logger.info(f"[UI] 用户选择: {feedback}")
    
    return JSONResponse({"status": "ok", "feedback": feedback})


@app.get("/api/sessions")
async def api_sessions(limit: int = 20):
    """获取会话列表"""
    store = get_session_store()
    sessions = store.list_sessions(limit=limit)
    return JSONResponse({"sessions": sessions})


@app.get("/api/session/{session_id}")
async def api_session(session_id: str):
    """获取单个会话详情"""
    store = get_session_store()
    session = store.get_session(session_id)
    if not session:
        return JSONResponse({"error": "会话不存在"}, status_code=404)
    
    # 解析 state_json
    try:
        state = json.loads(session.get("state_json", "{}"))
    except Exception:
        state = {}
    
    return JSONResponse({
        "session": session,
        "state": state,
    })


def start_ui_server(host: str = "127.0.0.1", port: int = 8080):
    """启动UI服务器"""
    logger.info(f"[UI] 启动控制面板: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    start_ui_server()
