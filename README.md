# 4智能体协作平台

基于 LangGraph + Kimi API 构建的多智能体协作系统，用于自动/半自动开发各种开源项目。

## 架构

```
┌─────────────────────────────────────────┐
│           人类（你）- 战略层               │
│   职责：市场分析、技能缺口判断、项目规划     │
│   工具：网页版AI（ChatGPT/Claude）辅助思考  │
└──────────────────┬──────────────────────┘
                   │ 提交：结构化项目需求
                   ▼
┌─────────────────────────────────────────┐
│     D - Orchestrator（编排调度中心）       │
│     框架：LangGraph StateGraph            │
│     职责：Milestone分解、状态管理、HITL    │
└──────────────────┬──────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐  ┌────────┐  ┌────────┐
│   B    │  │   E    │  │   C    │
│ 开发工程师│  │ 质量评估员│  │ 训练工程师│
└───┬────┘  └────────┘  └───┬────┘
    │                       │
    ▼                       ▼
┌────────┐            ┌────────┐
│Docker  │            │MLX/Mac │
│沙箱执行 │            │本地训练 │
└────────┘            └────────┘
```

## 智能体定义

| 智能体 | 职责 | 技术 | 输出 |
|--------|------|------|------|
| **D** | 编排调度 | LangGraph StateGraph | Milestone管理、Human-in-loop |
| **B** | 开发工程师 | Kimi API + Docker | 代码仓库、README、测试 |
| **E** | 质量评估员 | Kimi API | 评分报告、改进建议 |
| **C** | 训练工程师 | MLX / Colab | 微调模型、数据集 |

## 项目产出

B 开发的代码自动保存到 `projects/` 目录：

```
projects/
├── 01-langgraph-multi-agent/     # 多智能体协作系统
├── 02-mcp-tool-ecosystem/        # MCP工具生态
├── 03-agent-evaluation/          # Agent评估体系
└── ...更多项目
```

## 快速开始

```bash
# 1. 激活环境
source venv/bin/activate

# 2. 运行平台（开发指定项目）
python src/main.py "开发一个LangGraph多智能体系统"

# 3. 查看产出
ls projects/01-langgraph-multi-agent/
```

## 目录结构

```
resume-agent-system/
├── src/
│   ├── main.py              # CLI 主入口
│   ├── core/                # 平台核心代码
│   │   ├── D.py            # 编排器
│   │   ├── B.py            # 开发工程师
│   │   ├── E.py            # 质量评估员
│   │   ├── C.py            # 训练工程师
│   │   ├── llm.py          # LLM 客户端
│   │   ├── docker_sandbox.py # Docker 沙箱
│   │   ├── persistence.py  # SQLite 会话持久化
│   │   ├── shared_state.py # 共享状态
│   │   └── logger.py       # 日志系统
│   ├── db/                  # SQLAlchemy 数据层（SQLite/Postgres）
│   │   ├── base.py
│   │   ├── models.py
│   │   └── task_repository.py
│   ├── tasks/               # Celery 任务队列
│   │   ├── celery_app.py
│   │   ├── agent_task.py
│   │   └── task_store.py
│   └── ui/                  # 旧版 Python UI（已弃用）
├── api/                     # Go API Gateway + 新版静态 UI
│   ├── main.go
│   ├── static/
│   │   ├── index.html
│   │   ├── app.js
│   │   └── style.css
│   └── Dockerfile
├── projects/                # 项目产出
│   ├── 01-langgraph-multi-agent/
│   ├── 02-mcp-tool-ecosystem/
│   └── 03-agent-evaluation/
├── tests/                   # 平台级单元测试
├── scripts/                 # 本地启动/停止脚本
├── config/
│   └── config.yaml         # 平台配置
├── data/                   # 数据存储
├── logs/                   # 日志文件
└── README.md              # 本文件
```

## 状态流转

```
init → analysis → development → evaluation → training → human_review → done
         ↑______________________________________________|
```

- **analysis**: D 解析需求，创建项目定义
- **development**: B 编写代码，Docker 沙箱执行
- **evaluation**: E 评估质量，生成报告
- **training**: C 微调模型（可选）
- **human_review**: 人类审批（interrupt 节点）
- **done**: 流程完成

## 配置

```bash
cp .env.example .env
# 编辑 .env 填入 KIMI_API_KEY 等值
```

### Kimi Code API 格式选择

平台支持 Kimi Code 的两种 API 格式，通过 `.env` 中的 `KIMI_API_FORMAT` 切换：

- **OpenAI 兼容格式（推荐）**
  ```bash
  KIMI_BASE_URL=https://api.kimi.com/coding/v1
  KIMI_API_FORMAT=openai
  ```

- **Anthropic 兼容格式**
  ```bash
  KIMI_BASE_URL=https://api.kimi.com/coding/
  KIMI_API_FORMAT=anthropic
  ```

`src/core/llm.py` 会根据 `KIMI_BASE_URL` 自动推断格式；显式设置 `KIMI_API_FORMAT` 可覆盖自动推断。

## 测试

```bash
# 运行平台级单元测试（全部使用 Mock，不调用真实 API）
source venv/bin/activate
python -m pytest tests/ -v

# 运行 01 项目自带的单元测试
cd projects/01-langgraph-multi-agent
python -m pytest tests/ -v
```

## Phase 1 修复摘要

本次修复聚焦于让平台达到“可离线验证、可实际调用 API、数据不重复写入、沙箱基本安全”的最小可用状态：

1. **LLM 客户端**：支持 OpenAI / Anthropic 双格式，修复此前 endpoint 与 payload 不匹配的问题。
2. **SQLite 持久化**：修复 `save_session()` 重复写入 llm_usage / artifacts / evaluations / test_runs 的 bug。
3. **Docker 沙箱**：默认禁用容器网络、安装 `requirements.txt`、增加安全参数（no-new-privileges、cap-drop、read-only）。
4. **共享状态**：把 `status`、`error`、`logs` 补入 `SharedState` TypedDict，避免 LangGraph 丢弃这些字段。
5. **编排器结束状态**：流程成功完成后设置 `status="complete"`。
6. **平台级测试**：新增 `tests/test_llm_client.py`、`tests/test_persistence.py`、`tests/test_docker_sandbox.py`、`tests/test_orchestrator_routing.py`。

## Phase 2 Stage 1：高并发任务调度（Python + Go 混合架构）

已完成高并发改造，架构变为：

```
用户/前端 → Go API Gateway → Redis → Celery Worker (Python) → D/B/E/C
```

### 新增能力

1. **Go API Gateway**（`api/main.go`）
   - `POST /api/v1/tasks`：投递任务
   - `GET /api/v1/tasks/:id`：查询任务
   - `GET /api/v1/tasks/:id/events`：SSE 实时状态流
   - `POST /api/v1/tasks/:id/review`：提交人类审核反馈
   - 10 并发任务投递实测 53ms 内完成

2. **Celery + Redis 任务队列**（`src/tasks/`）
   - `Orchestrator.run()` 包装为 Celery Task
   - Worker 可水平扩展，支持多进程消费

3. **SQLAlchemy 数据层**（`src/db/`）
   - 默认 SQLite，可通过 `DATABASE_URL` 切换 Postgres
   - `tasks` 表持久化所有 API 投递的任务

4. **新版 Web UI**（`api/static/`）
   - 纯静态 HTML/JS，直接由 Go API 托管
   - 使用 SSE 实时接收任务状态，替换旧的 3 秒轮询

5. **异步 Human-in-the-Loop**
   - D 在人类审核节点进入等待状态
   - Go API 将反馈写入 Redis
   - Worker 读取反馈后继续执行

### 本地启动

```bash
# 方式一：一键脚本
./scripts/start_local.sh

# 方式二：手动分步
redis-server --daemonize yes
source venv/bin/activate
python -c "from src.db import init_db; init_db()"
celery -A src.tasks worker -l info -c 2 &
cd api && go build -o agent-api . && ./agent-api

# 打开 UI
open http://localhost:8080/
```

### Docker Compose（全栈）

```bash
docker-compose up --build
```

## Phase 2 Stage 2：向量数据库 + RAG

已完成 RAG 能力：

1. **向量存储**：Qdrant + `sentence-transformers/all-MiniLM-L6-v2`
2. **文档类型**：需求、代码、评估报告、项目摘要
3. **检索场景**：
   - **D 分析需求**：检索相似历史项目，复用技术栈/里程碑
   - **B 生成代码**：检索相似代码片段作为 in-context example
   - **E 评估代码**：检索相似评估报告，保持评分一致性
4. **自动索引**：任务完成后自动将结果写入向量库

## Phase 2 Stage 3：MCP 工具集成

已接入官方 MCP Python SDK，D/B/E 智能体现在可以通过统一协议调用外部工具：

1. **MCP Server**（`src/mcp/server.py`）：基于 FastMCP 暴露工具
2. **MCP Client**（`src/mcp/client.py`）：通过 stdio 连接 Server，对外暴露同步接口
3. **桥接层**（`src/mcp/bridge.py`）：`ask_with_tools` 支持 LLM 自动多轮工具调用
4. **内置工具**：
   - `file_read` / `file_write` / `file_list`：受限文件系统访问
   - `shell_exec`：Docker 沙箱执行命令，本地 fallback 带黑名单
   - `web_fetch`：只读 HTTP 获取
   - `vector_search`：检索历史任务记忆
5. **智能体集成**：
   - **D**：可联网调研、查看已有项目、查历史案例
   - **B**：可读取/写入文件、运行测试命令
   - **E**：可运行静态检查、深入检查代码
6. **开关**：通过 `MCP_ENABLED=true/false` 或 `config/config.yaml` 中的 `mcp.enabled` 控制

## Phase 2 Stage 4：企业级 ML 自动化评估

已构建面向大厂落地的 ML 评估体系：

1. **特征平台**（`src/ml/features.py`）：
   - 从 `static_metrics` + `test_results` 提取 22 维标准化特征
   - 特征版本化（`v1`），支持模型与特征模式解耦

2. **样本仓库**（`src/ml/dataset.py`）：
   - JSONL 存储，自动去重、数据血缘、版本快照
   - 每次任务评估后自动回流样本

3. **模型训练**（`src/ml/model.py`）：
   - `GradientBoostingRegressor` 预测 `overall_score`
   - `RandomForestClassifier` 预测 `passed`
   - 输出 MAE/RMSE/Accuracy/F1 验证报告

4. **模型仓库**（`src/ml/registry.py`）：
   - 与 MLflow 对齐的 stage：`shadow` / `staging` / `production` / `archived`
   - 支持注册、查询、升级、回滚

5. **评估服务**（`src/ml/service.py`）：
   - `shadow` 模式：仅记录 ML 分数，不改变决策
   - `online` 模式：与 LLM 评分加权融合（默认 0.7 LLM + 0.3 ML）
   - 降级能力：LLM 不可用时可直接使用 ML 评分

6. **监控**（`src/ml/monitoring.py`）：
   - 预测分布记录、简单漂移检测、Prometheus 指标输出

7. **数据飞轮**：
   - 任务完成后 Celery 自动触发 `train_ml_evaluator` 训练任务
   - 手动训练：`python -m src.ml.train --min-samples 20 --stage production`

## Phase 2 Stage 5：SFT 数据飞轮

已构建企业级 SFT 数据飞轮，把高质量任务轨迹转化为微调训练数据：

1. **多角色采集器**（`src/sft/collectors.py`）：
   - **D**：需求 → 项目架构
   - **B**：需求 + 架构 → 代码
   - **E**：代码 + 测试结果 → 评估报告
   - **C**：代码 → 函数级训练样本

2. **数据质量中心**（`src/sft/quality.py`）：
   - 过滤：通过测试、评分达标、无语法错误
   - 安全：API Key / 密码 / Token / PII 检测
   - 去重：基于 instruction-output 哈希

3. **数据集仓库**（`src/sft/dataset.py`）：
   - 多角色独立存储：`data/sft_datasets/{role}/v1/dataset.jsonl`
   - 版本快照、训练/验证拆分、统计指标

4. **格式转换器**（`src/sft/formats.py`）：
   - 支持 Alpaca、ShareGPT / OpenAI、MLX 格式

5. **训练流水线**（`src/sft/trainer.py`）：
   - MLX LoRA 脚本生成
   - 预留云端训练接口

6. **数据飞轮**（`src/sft/pipeline.py`）：
   - 任务完成后自动采集样本
   - 样本数达到阈值自动触发训练
   - C 智能体使用 SFTTrainer 生成训练脚本

7. **配置**：`SFT_ENABLED`、`SFT_MIN_SCORE`、`SFT_MIN_SAMPLES`、`SFT_ROLES`

## Phase 2 Stage 6：Go Gateway 生产化

已将单文件 Gin Gateway 重构为面向大厂生产环境的企业级 API Gateway：

1. **模块化结构**（`api/`）：
   - `config/`：环境变量配置
   - `middleware/`：认证、限流、日志、指标、恢复、CORS
   - `handlers/`：任务、事件、健康检查 handler
   - `service/`：业务逻辑
   - `store/`：Redis 封装

2. **认证与限流**：
   - API Key 认证（`Authorization: Bearer` 或 `X-API-Key`）
   - 令牌桶限流：全局限流 + 按 Key 限流

3. **可观测性**：
   - 结构化 JSON 日志 + `X-Request-ID` 链路追踪
   - Prometheus 指标：`/metrics`

4. **健康检查**：
   - `/livez`：存活探针
   - `/readyz`：就绪探针（依赖 Redis）
   - `/healthz`：综合健康

5. **稳定性**：
   - Panic 恢复中间件
   - 优雅关闭（`SIGINT` / `SIGTERM`）
   - SSE 事件流 heartbeat（15s）

6. **安全**：
   - 可配置 CORS
   - TLS/HTTPS 支持（`TLS_CERT` / `TLS_KEY`）
   - 统一错误响应格式

7. **测试**：
   - `go test ./...` 通过

## Milestone

- [x] Milestone 1: 平台环境搭建
- [x] Milestone 2: 接入 Kimi API，实现完整流程（Phase 1 完成基础接入与测试）
- [x] Milestone 3: 高并发任务调度（Phase 2 Stage 1 完成：Go API + Celery + Redis + SSE）
- [x] Milestone 4: 向量数据库 + RAG（Phase 2 Stage 2 完成：Qdrant + sentence-transformers）
- [x] Milestone 5: MCP 工具集成（Phase 2 Stage 3 完成：FastMCP + stdio + 内置工具）
- [x] Milestone 6: ML 自动化评估（Phase 2 Stage 4 完成：特征平台 + 模型仓库 + Shadow/Online 服务）
- [x] Milestone 7: SFT 数据飞轮（Phase 2 Stage 5 完成：多角色采集 + 质量中心 + 训练流水线）
- [x] Milestone 8: Go Gateway 生产化（Phase 2 Stage 6 完成：认证/限流/日志/指标/健康检查/优雅关闭）
- [ ] Milestone 9: 首个项目（LangGraph Multi-Agent）开发完成并达到可用质量

## License

MIT
