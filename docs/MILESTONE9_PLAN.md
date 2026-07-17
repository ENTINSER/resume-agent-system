# Milestone 9：通过智能体平台实际生成/迭代项目 01 的详细执行方案

> 保存时间：2026-06-30
> 状态：待执行（等待用户指令）
> 目标：使用平台级 D/B/E/C 工作流，对 `projects/01-langgraph-multi-agent` 进行自动化生成或迭代，并使其达到 README 声明的可用质量标准。

---

## 一、前置条件

### 1.1 必需的 API Key

- **1 个 Kimi API Key**：用于 Python Worker 调用 LLM。
  - 环境变量名：`KIMI_API_KEY`
  - 模型默认：`kimi-k2p5`
- **Go Gateway 的 API Key 可选**：
  - 本地测试建议不设置 `API_KEY`，关闭认证，方便 curl 调用。
  - 如需认证，再额外准备一个 `API_KEY` 环境变量。

### 1.2 本地环境

- Redis 已运行（`scripts/start_local.sh` 会启动）
- Docker 已安装（用于 B 开发工程师的沙箱测试）
- 已安装项目依赖：`source venv/bin/activate && pip install -r requirements.txt`

---

## 二、API 消耗估算

按当前平台默认模型 `kimi-k2p5` 计价（约 $0.5/1M input tokens，$1.5/1M output tokens）：

| 项目 | 估算 |
|------|------|
| 单次任务 LLM 调用次数 | 约 5–15 次（分析 + 开发 + 评估 + 可能的重写循环） |
| 单次任务 Token 消耗 | 输入约 10k–50k，输出约 5k–20k |
| 单次任务成本 | 约 **$0.5–$2（人民币 3–15 元）** |
| 最坏情况 | 若触发 `max_iterations=10` 且输出很长，可能 **$5–$10（人民币 35–70 元）** |

**建议策略**：先投递 1 个任务，观察实际消耗和输出质量，再决定是否继续。

---

## 三、旧项目是否会被替换

**默认不会**。当前 `agent_task.py` 只把结果写到 Redis/SQL，不会自动落盘到 `projects/`。

为了让生成结果可查看，需要小幅改动 `agent_task.py`，在任务完成后调用 `save_project()`。本方案采用**新目录名** `projects/01-langgraph-multi-agent-v2`，旧项目原封不动保留。

**兜底措施**：执行前会先备份旧项目：

```bash
cp -r projects/01-langgraph-multi-agent projects/01-langgraph-multi-agent-backup
```

---

## 四、需要做的代码改动

### 4.1 `src/tasks/agent_task.py`：支持自动化落盘

在任务完成后、返回 summary 之前，增加如下逻辑：

```python
from src.core.B import save_project

# ... 任务执行完成后 ...
if result.get("status") in ("complete", "completed"):
    try:
        save_project(result, "projects")
        logger.info(f"[Task] 项目已保存到 projects/")
    except Exception as e:
        logger.warning(f"[Task] 保存项目文件失败: {e}")
```

这样 Worker 会自动把生成的代码写入 `projects/{project_name}/`。

### 4.2 `src/tasks/agent_task.py`：关闭 UI 审核等待

当前 `agent_task.py` 中硬编码了：

```python
orchestrator = Orchestrator(ui_mode=True, task_id=task_id)
```

`ui_mode=True` 会在人类审核节点等待 Redis/文件输入，30 分钟无响应才会默认批准。

**改动**：通过环境变量 `AGENT_UI_MODE` 控制：

```python
import os
ui_mode = os.getenv("AGENT_UI_MODE", "true").lower() in ("true", "1", "yes")
orchestrator = Orchestrator(ui_mode=ui_mode, task_id=task_id)
```

执行时设置：

```bash
export AGENT_UI_MODE=false
```

即可实现全自动化，无需人工审核。

---

## 五、执行步骤

### 步骤 1：备份旧项目

```bash
cd /Users/mingrun/resume-agent-system
cp -r projects/01-langgraph-multi-agent projects/01-langgraph-multi-agent-backup
```

### 步骤 2：设置环境变量

```bash
export KIMI_API_KEY="你的 Kimi API Key"
export AGENT_UI_MODE="false"
# 如需关闭 Go Gateway 认证，则不设置 API_KEY
```

### 步骤 3：启动基础设施

```bash
scripts/start_local.sh
```

会启动：
- Redis（端口 6379）
- Celery Worker
- Go API Gateway（端口 8080）

### 步骤 4：投递任务

```bash
curl -X POST http://localhost:8080/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "requirements": "在 projects/01-langgraph-multi-agent-v2 目录下，实现一个 LangGraph 多智能体文章写作系统：包含 researcher、writer、reviewer 三个角色，能协作完成一篇文章；reviewer 从准确性、完整性、可读性、结构四个维度评分（满分100，通过阈值80），最多修改3次；要求代码可运行，pytest 测试通过，README 完整。"
  }'
```

返回示例：

```json
{
  "task_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "pending"
}
```

### 步骤 5：观察任务执行

**方式 A：SSE 实时观察**

```bash
curl http://localhost:8080/api/v1/tasks/{task_id}/events
```

**方式 B：查看 Worker 日志**

Worker 日志会输出各阶段状态、token 消耗、测试失败原因等。

**方式 C：查询任务状态**

```bash
curl http://localhost:8080/api/v1/tasks/{task_id}
```

### 步骤 6：验收结果

任务完成后，检查生成目录：

```bash
ls -la projects/01-langgraph-multi-agent-v2/
```

运行其测试：

```bash
cd projects/01-langgraph-multi-agent-v2
source ../../venv/bin/activate
pytest tests/ -v
```

### 步骤 7：收尾

- 如果生成结果满意：
  - 可选择把 `01-langgraph-multi-agent-v2` 重命名为 `01-langgraph-multi-agent` 进行替换
  - 或保留两个版本，更新 `projects/README.md` 状态
- 如果不满意：
  - 再投递一次任务，要求基于 `v2` 继续改进
  - 或调整 requirements 后重新生成

---

## 六、风险与兜底

| 风险 | 可能性 | 兜底措施 |
|------|--------|---------|
| 生成代码质量差、测试不通过 | 中 | 平台最多迭代 10 次后自动结束；可再投递新任务要求重写 |
| API 消耗超预期 | 中 | 先跑 1 个任务观察实际 token/cost，再继续 |
| 覆盖旧项目 | 低 | 输出到 `v2` 目录，且已备份旧项目 |
| 人类审核等待 30 分钟 | 低 | 已通过 `AGENT_UI_MODE=false` 关闭 |
| Docker 沙箱运行失败 | 低 | 平台会降级为 mock 测试，但会记录日志 |
| Redis/Worker 启动失败 | 低 | 检查 `scripts/start_local.sh` 日志，手动 `docker-compose up -d` |

---

## 七、验收标准

完成本方案后，应满足：

- [ ] `projects/01-langgraph-multi-agent-v2/` 下包含完整项目文件
- [ ] 项目可通过 `pytest tests/ -v`（至少核心测试通过）
- [ ] README 包含使用说明、架构说明、评估标准
- [ ] `projects/README.md` 中项目 01 状态更新为“已完成”

---

## 八、备注

- 本方案为**自动化无人工审核**版本，适合批量迭代。
- 如果希望保留人类审核节点，可设置 `AGENT_UI_MODE=true` 并通过 Web UI 或 Redis 提交反馈。
- 实际执行前，请确保 `KIMI_API_KEY` 可用且余额充足。
