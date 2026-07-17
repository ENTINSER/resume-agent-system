# 4智能体平台开发进度

> 本文档用于替代长对话中的上下文记忆，每次会话开始时请先读取此文件。

## 当前阶段

**Phase 6：端到端项目落地验证（Milestone 9）**

- 启动 Redis + Celery Worker + Go API 本地服务
- 修复 Go API 投递 Celery 任务的消息格式兼容性（`properties`/`headers`/`body` 信封）
- 通过 `/api/v1/tasks` 提交首个真实项目生成任务
- 观察 D-B-E-C 编排、RAG 检索、自动迭代与项目落盘

## 已完成的里程碑

### Phase 1：基础修复与测试
- [x] LLM 客户端支持 OpenAI / Anthropic 双格式
- [x] 修复 SQLite 重复写入 bug
- [x] Docker 沙箱安全加固（默认禁网、安装 requirements、安全参数）
- [x] 补全 SharedState 字段（status/error/logs）
- [x] 平台级单元测试全部通过

### Phase 2 Stage 1：高并发架构
- [x] 安装 Celery、Redis、SQLAlchemy、psycopg2、Go
- [x] `Orchestrator.run()` 包装为 Celery Task
- [x] TaskStore（Redis + SQL 双写）+ TaskRepository
- [x] Go API Gateway（任务投递/查询/SSE/人类反馈）
- [x] 新版静态 Web UI（SSE 实时更新）
- [x] 异步 Human-in-the-Loop（Redis 反馈通道）
- [x] docker-compose.yml + Dockerfile.worker + api/Dockerfile
- [x] 本地启动脚本 `scripts/start_local.sh` / `scripts/stop_local.sh`
- [x] 集成测试：`tests/test_integration_queue.py`
- [x] 并发测试：10 任务并发投递 53ms 内完成

### Phase 2 Stage 2：向量数据库 + RAG
- [x] 引入 Qdrant + sentence-transformers
- [x] 创建 `src/vector_store/` 模块
- [x] D/B/E 相似案例检索
- [x] 任务完成时索引到向量库
- [x] 更新 docker-compose、启动脚本、配置

### Phase 2 Stage 3：MCP 工具集成
- [x] 工具封装为 MCP Server（`src/mcp/server.py`）
- [x] Agent 通过 MCP Client 调用工具（`src/mcp/client.py`）
- [x] 桥接层 `ask_with_tools` 支持 LLM 自动多轮工具调用（`src/mcp/bridge.py`）
- [x] 内置工具：file_read / file_write / file_list / shell_exec / web_fetch / vector_search

### Phase 2 Stage 4：企业级 ML 自动化评估
- [x] 特征平台（`src/ml/features.py`）22 维标准化特征
- [x] 样本仓库（`src/ml/dataset.py`）JSONL 存储、去重、版本快照
- [x] 模型训练（`src/ml/model.py`）回归 + 分类双模型
- [x] 模型仓库（`src/ml/registry.py`）shadow / staging / production / archived
- [x] 评估服务（`src/ml/service.py`）shadow / online / disabled 三模式
- [x] 监控（`src/ml/monitoring.py`）预测分布、漂移检测、Prometheus 指标
- [x] 数据飞轮：任务完成后自动触发 `train_ml_evaluator`

### Phase 2 Stage 5：SFT 数据飞轮
- [x] 多角色采集器（`src/sft/collectors.py`）D/B/E/C
- [x] 数据质量中心（`src/sft/quality.py`）过滤、安全、去重
- [x] 数据集仓库（`src/sft/dataset.py`）多角色独立存储、版本快照
- [x] 格式转换器（`src/sft/formats.py`）Alpaca / ShareGPT / MLX
- [x] 训练流水线（`src/sft/trainer.py`）MLX LoRA 脚本生成
- [x] 数据飞轮（`src/sft/pipeline.py`）自动采集、阈值触发训练

### Phase 2 Stage 6：Go Gateway 生产化
- [x] 模块化结构：config / middleware / handlers / service / store
- [x] API Key 认证 + 令牌桶限流（全局 + 按 Key）
- [x] 结构化 JSON 日志 + `X-Request-ID`
- [x] Prometheus 指标 `/metrics`
- [x] 健康检查 `/livez` / `/readyz` / `/healthz`
- [x] Panic 恢复 + 优雅关闭 + SSE heartbeat
- [x] CORS + TLS/HTTPS 支持
- [x] `go test ./...` 通过

### Phase 5：RAG 数据库建设
- [x] 扩展向量模型支持的 `doc_type` 集合
- [x] 嵌入服务 LRU 缓存 + 大批量编码修复 + 缓存全命中 KeyError 修复
- [x] Indexer 确定性 ID、滑动窗口分片、README/requirements/测试结果索引
- [x] 外部仓库种子：keon/algorithms、psf/requests、cosmicpython/code、TheAlgorithms/Python
- [x] 历史会话/项目目录索引
- [x] 合成示例目录与默认兜底
- [x] 检索策略：混合排序、rerank、多跳检索、失败模式检索
- [x] Qdrant 本地持久化回退（`data/qdrant_storage`）

### Phase 6：端到端项目落地验证（进行中）
- [x] 修复 Go API 投递 Celery 任务的消息格式兼容性
- [x] 修复 macOS 下 Celery Worker fork + MPS 崩溃问题
- [x] 首个真实项目生成任务已提交并运行中
- [ ] 等待任务完成/失败并验收生成结果

## 测试状态

```bash
# Python 平台核心单元测试
source venv/bin/activate
python -m pytest tests/test_llm_client.py tests/test_persistence.py tests/test_orchestrator_routing.py tests/test_task_store.py tests/test_db.py tests/test_mcp_server.py tests/test_mcp_client.py tests/test_mcp_prompts.py tests/test_docker_sandbox.py tests/test_logger.py tests/test_exceptions.py tests/test_metrics.py tests/test_smoke_project01.py -v
# 结果：88 passed

# 项目 01 自身测试
cd projects/01-langgraph-multi-agent
python -m pytest tests/ -v
# 结果：27 passed

# Go API 构建与测试
cd api && go test ./... -v
# 结果：全 pass
```

## 待做工作（按优先级）

### Milestone 9：首个项目落地
- [x] 通过平台实际运行生成/迭代 `projects/01-langgraph-multi-agent`
- [x] 确保项目 01 达到 README 中声明的可用质量标准（本地测试 27 passed）
- [x] 更新 `projects/README.md` 中项目 01 状态为“已完成”

### 工程化加固（公司级可落地）
- [x] 关闭 MCP `shell_exec` 本地 fallback，消除安全隐患
- [x] 修复 Go API `GetTask` result 字符串未正确反序列化为对象的问题
- [x] Orchestrator 全阶段事件统一发布到 Redis，补齐 SSE 中间状态
- [x] Docker 沙箱测试策略：read_only 与 pip 安装冲突的解决方案
- [x] Celery 任务幂等、超时、重试机制完善
- [x] 统一配置层（Pydantic Settings），结束 `.env` / `yaml` / `os.getenv` 混用
- [x] Python 端结构化 JSON 日志，与 Go Gateway 日志格式对齐
- [x] 数据库：Alembic 迁移、Postgres 生产默认、`datetime.utcnow()` 替换
- [x] 异常分级：定义可降级错误 vs 致命错误
- [x] 任务取消 API：`POST /api/v1/tasks/:id/cancel`
- [x] API 配额与审计（按 API Key 统计 token/cost）
- [x] Python 端 Prometheus 指标暴露
- [x] CI/CD + API 集成测试
- [x] K8s 部署清单补全

## 关键文件路径

| 用途 | 路径 |
|------|------|
| LLM 客户端 | `src/core/llm.py` |
| 编排器 | `src/core/D.py` |
| 开发工程师 | `src/core/B.py` |
| 质量评估员 | `src/core/E.py` |
| Celery 任务 | `src/tasks/agent_task.py` |
| 任务状态存储 | `src/tasks/task_store.py` |
| 数据层 | `src/db/base.py`, `src/db/models.py` |
| MCP Server | `src/mcp/server.py` |
| 向量存储 | `src/vector_store/vector_store.py` |
| ML 评估服务 | `src/ml/service.py` |
| SFT 飞轮 | `src/sft/pipeline.py` |
| Go API | `api/main.go` |
| 新版 UI | `api/static/index.html`, `api/static/app.js` |
| 启动脚本 | `scripts/start_local.sh` |
| Docker Compose | `docker-compose.yml` |
| 测试 | `tests/` |
