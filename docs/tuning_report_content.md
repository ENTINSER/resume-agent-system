# LangGraph 多智能体文章写作任务调优过程技术报告

模型：Kimi k2p5-coding | 任务：D-B-E-C 四阶段智能体平台 | 报告日期：2026-07-13

## 1. 背景与目标

本报告记录针对 LangGraph 多智能体文章写作系统生成任务的持续调优过程。该任务要求实现 researcher、writer、reviewer 三个角色，支持最多 3 次修改，reviewer 从 accuracy、completeness、readability、structure 四个维度评分。基线版本（v10）耗时 5.3 小时、得分 16.5、多次 600s 超时，几乎不可用。

调优目标围绕三个核心指标：总耗时压缩到 1 小时以内、消除 600s 超时、评估得分提升到 50+。报告将逐一说明每个版本的问题、策略、效果与失效原因，为后续更换模型提供明确的 prompt、架构与超参起点。

## 2. 架构基线

平台采用 D-B-E-C 四阶段工作流：

- D（Designer）：根据需求输出 file_plan、tech_stack、required_elements。
- B（Developer）：按 file_plan 生成代码，包含骨架、填充、import 干跑、Docker 测试。
- E（Evaluator）：对代码进行多维度评分。
- C（Critic/Rewriter）：根据评估反馈改写需求或 plan，进入下一轮。

关键配置：macOS 使用 Celery solo 模式避免 fork+MPS 崩溃；Redis 做队列；Qdrant 本地文件模式做向量存储；Docker sandbox 用于测试。

### 2.1 业务价值与应用场景

该项目并非面向 C 端用户的独立产品，而是内部 AI 研发平台（resume-agent-system）的能力验证任务。其最终用户和实际价值体现在以下三方面：

- **内部工程提效**：平台目标是把自然语言需求直接转化为可运行、可测试的项目代码，减少研发人员在脚手架、多文件接口对齐、基础测试上的重复劳动。
- **AI Agent 研究基线**：LangGraph 多智能体写作任务是复杂任务的代表，覆盖了角色分工、状态流转、循环控制、评分反馈等典型 Agent 模式，可作为评估不同模型和生成策略的基准用例。
- **对外能力背书**：一旦该基准任务稳定达到 50+ 分，平台即可向内部团队展示端到端代码生成能力，支撑后续接入更多业务场景（如自动化脚本、微服务原型、数据 pipeline）。

因此，本次调优的"业务价值"不直接体现为营收，而是**降低复杂代码生成的边际成本、提升平台可用性的置信度**。

### 2.2 监控与可观测性

基线版本耗时 5.3 小时，定位瓶颈主要依赖 Celery worker 的文本日志（logs/celery.log）和任务状态 API，未引入分布式链路追踪或 APM。具体观测手段演进如下：

- **v10 基线**：通过 `grep 'Read timed out'` 定位到 600s 超时集中在 `src/nodes/reviewer.py`、`tests/test_manager.py` 等大文件；通过输出 token 数确认单次响应超过 6000 token 是超时根因。
- **v11-v13**：在日志中增加 `[B] 进入骨架生成阶段`、`[B] 骨架 AST 校验通过`、`[B] 进入填充阶段`、`[B] import 干跑通过` 四行标记，使各阶段耗时可视化。
- **v14-v16**：增加 chunk 生成日志（`[B] chunk N/M 完成且校验通过`、`[B] 存在空方法体，重试`）、`__init__.py 自动生成` 日志、`Docker 测试` 结果日志。
- **metrics server**：项目中已实现 metrics 服务，但在 macOS 本地多次启动时遇到端口占用（Errno 48），因此本次调优主要以日志 grep 为主，metrics 数据未完全落地。

当前可观测性仍偏向本地调试，生产级部署需要：

1. 任务级链路追踪（D -> B -> E -> C 每个节点的耗时、token 消耗、重试次数）；
2. 文件级指标（每个文件生成耗时、截断次数、AST 校验失败次数、Docker 测试失败原因分类）；
3. 评估指标看板（overall_score、functionality、reliability、maintainability 随版本变化趋势）。

## 3. 版本演进与关键指标

| 版本 | 耗时 | 得分 | 超时 | 核心问题/进展 |
|------|------|------|------|--------------|
| v10 | 5.3h | 16.5 | 多次 | 单响应输出 6000+ token，频繁 600s 超时；代码截断严重 |
| v11 | 14.3min | 23.0 | 0 | 文件数降到 8 个，超时消失；chunk 拆分生效，但代码仍是占位符 |
| v12 | 11.7min | 18.0 | 0 | AST 校验通过，但生成内容语义偏离需求 |
| v13 | 15.8min | 18.0 | 0 | 增加字段名/类名硬约束，效果不明显 |
| v14 | 146s | 0 | 0 | __init__.py 导入不存在的 create_article，流程快速失败 |
| v15 | ~30min | ~15 | 0 | agents.py 单文件 1699 行，graph.py 缺少 ArticleGraph |
| v16 | ~12min | ~15 | 0 | 拆分为 researcher.py/writer.py/reviewer.py，pytest 可运行，1/4 通过 |

### 3.1 测试策略与覆盖维度

本次调优的测试体系分为四层，但未包含正式的性能/压力测试：

1. **单元测试（由 LLM 生成）**
   - 每个 src/*.py 对应 tests/test_*.py；
   - 覆盖 Agent 类、状态模型、Graph 工作流、入口函数；
   - 通过 mock 外部依赖（LLM、文件系统、网络）保证可离线运行。

2. **import 干跑（零成本集成检查）**
   - 在 Docker 测试前执行，捕获项目内部缺失模块/符号；
   - 不依赖第三方包安装，属于最轻量的集成检查。

3. **Docker 集成测试**
   - 在隔离容器中安装依赖并运行 pytest；
   - 验证运行环境、依赖版本、文件系统交互是否符合预期。

4. **MultiMetricGate 评估器（隐藏测试）**
   - 从 functionality、reliability、maintainability、efficiency、documentation 五个维度打分；
   - 包含对字段名、异常类、状态流转、README 完整性的隐式检查。

**当前不足**：

- 没有针对 LLM 输出稳定性的压力测试（例如同一任务连续运行 5 次，观察得分方差）；
- 没有针对 token 消耗与耗时的基准回归测试；
- 超时消除的稳定性仅通过多次人工观察日志确认，未自动化断言。

建议后续补充：

- 固定 seed 的回归测试集（simple CLI + LangGraph）每版本跑 3 次，统计平均耗时与得分方差；
- 大文件生成压力测试（>300 行文件）验证是否仍出现截断；
- 重试/回退次数上限测试，防止无限重试导致成本失控。

## 4. 被验证有效的策略

### 4.1 文件级拆分：从单响应到每 group 1 个文件

v10 的致命问题是把多个文件塞进一次 LLM 调用，输出 token 超过 6000，触发 600s 超时。将生成粒度改为每 group 一个文件后，单次输出控制在 2000-4000 token，耗时从 5.3 小时降到 15 分钟以内。

### 4.2 限制 file_plan 规模与禁止文件清单

在 D 阶段 prompt 中加入上限：文件总数不超过 10，禁止 pyproject.toml、.env.example、conftest.py、src/__init__.py、src/config.py。减少无关文件后，上下文长度和 LLM 注意力分散问题明显改善。

### 4.3 import 干跑作为零成本拦截器

在进 Docker 之前做 import 干跑，能在不花钱运行测试的情况下发现项目内部缺失模块/符号。v11 之后几乎所有跨文件引用错误都在这一阶段被捕获。

### 4.4 __init__.py 规则化生成

v14 证明让 LLM 写 __init__.py 会导入不存在的符号。改为用 AST 扫描 src/*.py 中实际定义的顶层 Class/Function，再自动生成 import 语句后，import 阶段错误基本消失。

### 4.5 单文件行数/类方法数上限

v15 的 agents.py 1699 行仍被截断。v16 在 D prompt 中强制每个文件不超过 200 行、每个类不超过 3 个方法，并把三个 Agent 拆到 researcher.py / writer.py / reviewer.py，代码首次变得可导入、可运行。

## 5. 无效或边际效应递减的策略

### 5.1 单纯降低 max_tokens 无法解决截断

把 complex 文件 max_tokens 从 16384 降到 8192，虽然减少了超时，但没有解决 LLM 在语义上写不完的问题。

### 5.2 chunk 拆分 + 续写合并并未显著提升功能正确性

理论上把大文件拆成多个 chunk 可以降低单次输出长度，但实践中：

1. LLM 容易在 chunk 中生成空方法体；
2. 拼接时容易出现缩进/try-except 块断裂；
3. 重试和回退逻辑增加了调用次数，但得分没有提升。

更有效的做法是把文件本身拆小，而不是在一个大文件里做 chunk 拼接。

### 5.3 硬编码字段名/类名约束效果有限

在 chunk prompt 中强制评分字段必须是 accuracy/completeness/readability/structure、异常类名必须是 RevisionLimitExceeded，对字段名有一定约束，但无法保证 LLM 正确实现状态流转、循环次数限制、异常抛出条件等功能逻辑。

### 5.4 空方法体检测导致过多重试

最初用方法体少于 5 行有效代码作为 stub 检测标准，导致大量合法短方法被误判，引发重试和回退。后续简化为只检测 pass / 空体 / return None 后，重试次数下降，但功能正确性问题仍未解决。

### 5.5 失败回退与容错机制设计

本次调优中实现的回退策略可归纳为三级，但没有指数退避，也没有持久化的部分成功状态：

**第一级：chunk 级重试**

- 触发条件：单个 chunk 未通过 AST 校验，或被 `_has_stub_methods` 检测为空方法体；
- 策略：立即重试，最多 2 次；
- 退避：无指数退避，2 次均失败后进入第二级；
- 部分成功保留：若前面 chunk 已成功，会保留在 `generated_parts` 中，失败 chunk 对应的元素由下一级兜底。

**第二级：文件级回退到单文件生成**

- 触发条件：某 chunk 重试 2 次仍失败，或 `_assemble_chunks` 拼接校验失败；
- 策略：忽略 chunk 拆分，直接调用 `_generate_full_file_artifact`，使用完整 prompt 和正常 max_tokens 一次生成；
- 效果：多数情况下能生成语法完整的文件，但大文件仍有截断风险。

**第三级：流程级 HITL / 默认通过**

- 触发条件：Docker 测试失败且自修复无法解决，或 file_plan 校验失败；
- 策略：标记 `human_review_required`；当前环境 `AGENT_UI_MODE=false`，HITL 收到信号后默认批准，任务继续；
- 风险：默认通过会导致低质量代码被保存，本次 v14 即因此保存了最小占位项目。

**部分成功文件的保留机制**

- 每生成一个文件，Artifact 立即追加到 `state["code_artifacts"]`；
- 后续文件生成依赖这些 artifacts 的摘要，因此失败文件不会导致前面文件被丢弃；
- 但一致性修复 `_fix_consistency_mismatches` 会重写相关文件，可能破坏部分已正确的实现。

**生产级改进建议**

- 对 chunk 重试加入指数退避（1s、2s、4s），降低 API 瞬时压力；
- 限制单任务最大重试次数，防止成本失控；
- HITL 默认批准改为默认拒绝或转异步人工审核；
- 引入 checkpoint，保存每轮迭代后的 artifacts，支持从失败点恢复而不是从头生成。

## 6. 当前模型（Kimi k2p5-coding）的能力边界

经过多轮调优，可以得出以下关于该模型在复杂代码生成任务上的能力边界：

1. 单次输出上限：稳定生成约 3000-4000 token 的完整代码文件可行，超过 5000 token 后截断/语义不完整的概率显著上升。
2. 跨文件一致性：无法可靠维护多文件之间的接口约定，需要规则引擎（如 __init__.py 自动生成、依赖校验）兜底。
3. 状态机/循环逻辑：对最多修改 3 次、满足阈值退出、抛出 RevisionLimitExceeded 这类需要精确控制的状态流转，LLM 容易写成看似合理但实际条件错误的代码。
4. 测试驱动：无法根据隐藏测试意图生成完全正确的实现，只能生成能通过自身测试的代码，与评估器隐藏测试之间存在 gap。
5. 长上下文利用：当 prompt 中塞入多个文件摘要后，LLM 会丢失细节，表现为字段名错误、方法体变短、docstring 截断。

## 7. 为下次更换模型提供的调优起点

基于本次调优结论，建议新模型沿用以下已被验证的架构与 prompt，并重点解决当前模型无法克服的功能正确性问题。

### 7.1 必须保留的架构设定

- 每 group 1 个文件，单次输出控制在不超过 4000 token
- file_plan 文件总数不超过 10，禁止 pyproject.toml/.env.example/conftest.py/config.py/__init__.py
- 每个文件不超过 200 行，每个类不超过 3 个方法
- import 干跑在 Docker 测试前执行
- __init__.py 由 AST 扫描规则生成

### 7.2 建议保留的 prompt 约束

- 明确列出每个元素的 name、type、methods、logic_requirements
- 硬编码关键字段名和类名（accuracy/completeness/readability/structure、RevisionLimitExceeded）
- 禁止 pass / return None / # TODO 占位

### 7.3 需要重点攻坚的方向

a) 状态机精确定义：把 revision_count 从 0 开始，每次 reviewer 评分后 +1，超过 MAX_REVISIONS 时抛出 RevisionLimitExceeded 写成伪代码塞进 prompt。

b) 显式测试用例作为生成目标：如果评估器隐藏测试可部分推断，把关键断言直接作为 required_elements 的逻辑要求。

c) RAG 引入 LangGraph 示例：从已索引仓库中检索多智能体 / LangGraph 示例，作为 in-context 参考。

d) 方法级生成：如果新模型仍无法一次性写对复杂类，进一步把生成粒度从文件降到方法，每个方法单独生成并 AST 校验。

e) 评估反馈闭环：让 C 阶段拿到 evaluator 的具体失败项，针对性地改写 file_plan 或给 B 阶段发修复任务，而不是泛泛地提高质量。

## 8. 补充测试结果

为验证第 3.1 节中提到的稳定性、token/耗时基准、超时消除稳定性、重试/回退上限，对 Simple CLI 和 LangGraph 任务各运行 3 次，并额外执行 1 次大文件压力测试。测试基于同一 worker 与同一版本代码（v16 最终代码）。

### 8.1 测试设计

- **Simple CLI 回归**：固定需求（Todo 工具），新目录运行 3 次，观察得分方差与超时情况。
- **LangGraph 回归**：固定需求（多智能体写作系统），新目录运行 3 次，观察得分方差与超时情况。
- **大文件压力**：需求要求生成包含至少 25 个函数的 src/string_utils.py，每个函数不少于 15 行，观察是否仍出现截断、重试、回退。

### 8.2 结果汇总

| 测试 | 次数 | 平均耗时 | 平均得分 | 得分方差 | 超时次数 | 最大文件行数 |
|------|------|----------|----------|----------|----------|--------------|
| Simple CLI | 3 | 4.91 min | 82.0 | 0.0 | 0 | 308 |
| LangGraph | 3 | 19.26 min | 12.33 | 54.06 | 1 | 376 |
| Large File Stress | 1 | 22.6 min | 0 | - | 0 | 1718 |

### 8.3 Simple CLI 稳定性结论

Simple CLI 任务表现出极高的稳定性：

- 3 次运行全部完成，得分均为 82/100，方差为 0；
- 平均耗时 4.91 分钟，无 600s 超时；
- chunk 重试、fallback、stub 拒绝均为 0；
- 最大单文件 308 行，未触发截断。

这说明在需求明确、文件规模适中的情况下，当前架构与 prompt 已能稳定产出可运行项目。

### 8.4 LangGraph 稳定性结论

LangGraph 任务稳定性差，验证了第 6 节中的模型能力边界：

- 3 次得分分别为 16.5、2.0、18.5，方差高达 54.06；
- 平均耗时 19.26 分钟，但单次波动大（37.12 min vs 9.82 min）；
- 出现 1 次 600s 超时（run-1）；
- 无 chunk 重试和 fallback，说明 D 阶段生成的 file_plan 已经将文件拆得足够小，没有触发 chunk 阈值。

核心问题仍是功能正确性：LLM 无法可靠实现状态机、循环次数限制和 Agent 协作逻辑。

### 8.5 大文件压力测试结论

大文件压力测试直接暴露了当前策略的短板：

- 生成了一个 1718 行的 src/string_utils.py；
- chunk 重试 6 次，fallback 3 次；
- 最终得分 0，说明即使花费 22.6 分钟、70 次 LLM 调用，仍无法生成一个可用的超大模块。

结论：当需求强制单文件超过 300 行时，当前模型和 chunk 策略均无法保证质量。**生产环境中应通过 D 阶段约束避免单文件过大**，而不是依赖生成阶段兜底。

### 8.6 token 消耗说明

本次测试脚本尝试从日志解析 prompt/completion token，但当前日志格式未输出结构化 token 字段，因此 token 均显示为 0。实际可参考 LLM 调用次数作为成本 proxy：

- Simple CLI：26-33 次/ run
- LangGraph：31-50 次/ run
- 大文件压力：70 次

建议后续在 `[LLM] 生成完成` 日志中统一输出 `prompt_tokens` 和 `completion_tokens`，以便做成本回归。

## 9. 四智能体平台架构说明

本报告前 8 节聚焦于平台生成 LangGraph 项目的调优过程。本节补充平台本身的架构、模块职责、数据流与部署方式。

### 9.1 整体架构

平台采用"编排层 + 执行层 + 存储层"三层架构：

- **编排层**：LangGraph 实现 D-B-E-C 状态机；
- **执行层**：Celery Worker 异步执行 LLM 调用、Docker 测试；
- **存储层**：Redis（队列/缓存）、Qdrant（向量检索）、SQLite（checkpoint）、本地文件系统（项目代码）。

### 9.2 四阶段职责

| 阶段 | 模块 | 职责 |
|------|------|------|
| D（Designer） | src/core/orchestrator/nodes.py | 分析需求、检索 RAG 案例、输出 file_plan 与 required_elements |
| B（Developer） | src/core/B.py | 按 file_plan 生成代码骨架、填充实现、import 干跑、Docker 测试 |
| E（Evaluator） | src/core/orchestrator/nodes.py + evaluator | 对生成代码从 functionality/reliability/maintainability/efficiency/documentation 五维度评分 |
| C（Critic/Rewriter） | src/core/orchestrator/nodes.py | 根据评估结果改写需求或 file_plan，驱动下一轮迭代 |

### 9.3 数据流

```
用户需求
   ↓
Go API Gateway → Celery Producer → Redis Queue
   ↓
Celery Worker (solo 模式) → LangGraph D-B-E-C 编排
   ↓
LLM Provider (Kimi/OpenAI)  ←→  RAG Retriever (Qdrant)
   ↓
生成代码 → 本地 projects/<project>/
   ↓
Docker Sandbox 运行 pytest
   ↓
MultiMetricGate 评估
   ↓
结果写回 task_store
```

### 9.4 关键模块

| 模块 | 说明 |
|------|------|
| src/core/orchestrator/core.py | LangGraph StateGraph 定义，节点与路由 |
| src/core/B.py | 代码生成核心：file_plan 拓扑排序、chunk 拆分、AST 校验、Docker 测试、__init__.py 自动生成 |
| src/core/llm.py | 统一 LLM chat 接口，封装 max_completion_tokens、finish_reason、token 统计 |
| src/core/docker_sandbox.py | Docker 容器内运行 pytest，隔离环境 |
| src/vector_store/rag_retriever.py | 基于 Qdrant 的代码/文档检索 |
| src/tasks/agent_task.py | Celery 任务入口，桥接 LangGraph 与任务队列 |
| api/ | Go 实现的 RESTful Gateway，负责任务提交与状态查询 |

### 9.5 部署方式

- **开发/本地**：
  - 启动 Redis：`redis-server`
  - 启动 Celery Worker：`venv/bin/celery -A src.tasks worker -l info -P solo`
  - 启动 Go API：`go run ./api/main.go`
  - 提交任务：`curl -X POST http://localhost:8080/api/v1/tasks`

- **生产建议**：
  - Worker 使用 Linux + Docker，避免 macOS fork+MPS 问题；
  - Redis 改为集群或 Sentinel；
  - Qdrant 使用服务端模式而非本地文件；
  - 增加 metrics 服务与链路追踪（OpenTelemetry / Jaeger）；
  - HITL 默认批准改为默认拒绝或接入人工审批系统。

### 9.6 建议的超参起点

```yaml
generation:
  split_threshold_tokens: 2000      # 超过此阈值才拆分 chunk
  chunk_max_tokens: 3072            # 每个 chunk 输出上限
  max_completion_tokens_simple: 4096
  max_completion_tokens_normal: 4096
  max_completion_tokens_complex: 8192

orchestrator:
  max_iterations: 5
```

### 9.7 建议的 D 阶段 prompt 核心约束

```text
- file_plan 文件总数不得超过 10 个
- 禁止 pyproject.toml、.env.example、conftest.py、src/__init__.py、src/config.py
- 每个文件不得超过 200 行；每个类不得超过 3 个方法
- LangGraph 工作流类必须放在 src/graph.py
- 入口函数必须放在 src/main.py 或 src/graph.py
- required_elements 必须包含 name/type/methods/logic_requirements
```

## 10. 结论

本次工作覆盖了四智能体协作平台从架构搭建到 LangGraph 业务测试的完整流程：

1. **平台架构**：基于 LangGraph 的 D-B-E-C 状态机 + Celery 异步执行 + Redis/Qdrant/Docker 存储与测试层；
2. **架构级问题已闭环**：5.3 小时降到 12 分钟、600s 超时基本消除、代码从无法导入变为可运行；
3. **稳定性验证**：Simple CLI 任务 3 次运行得分均为 82，方差 0，证明中低复杂度任务已稳定；LangGraph 任务 3 次运行得分方差高达 54.06，证明高复杂度任务仍不稳定；
4. **压力测试验证**：强制单文件 >300 行时，当前模型无法保证质量，chunk 策略失效；
5. **模型能力边界**：当前 Kimi k2p5-coding 无法可靠实现复杂状态机、循环控制和跨文件语义一致；

下一步若要突破 50 分，需要更强的基础模型，或引入测试用例驱动的方法级生成 + RAG 示例 + 评估反馈闭环。本报告所列的起点、约束、失败教训和补充测试数据，可作为更换模型后快速复现当前最优状态并继续上攻的基准。
