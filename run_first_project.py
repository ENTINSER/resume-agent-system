#!/usr/bin/env python3
"""启动首个项目：LangGraph Multi-Agent 协作系统"""

import sys
sys.path.insert(0, '/Users/mingrun/resume-agent-system')

from dotenv import load_dotenv
load_dotenv()

from src.core.D import orchestrator

requirements = """# 项目：LangGraph Multi-Agent 协作系统

## 目标
构建一个基于 LangGraph 的多智能体协作平台，支持研究员、写手、审核员三个角色协作完成文章/报告生成任务。

## 架构要求

### 1. 智能体角色（3个）
- **研究员 (Researcher)**: 收集信息、分析数据、提取关键点
- **写手 (Writer)**: 基于研究成果撰写内容
- **审核员 (Reviewer)**: 质量检查、反馈修改建议

### 2. 状态图 (StateGraph)
使用 LangGraph 构建状态流转：
```
input -> research -> draft -> review -> [approve] -> output
                    -> [reject] -> revise -> review
```

### 3. 共享状态
```python
class WorkflowState(TypedDict):
    topic: str              # 用户输入主题
    research_notes: str     # 研究成果
    draft: str              # 草稿内容
    review_feedback: str  # 审核反馈
    revision_count: int     # 修改次数
    approved: bool          # 是否通过
```

### 4. LLM 后端
支持两种模式：
- OpenAI API（默认）
- 本地 MLX 模型（可选，通过环境变量切换）

### 5. 配置文件
使用 YAML 配置文件管理：
- LLM 参数（模型、温度、max_tokens）
- 各智能体提示词模板
- 审核标准（评分阈值）
- 最大修改次数

### 6. 命令行接口
```bash
python main.py --topic "AI发展趋势" --config config.yaml --output report.md
```

### 7. 测试要求
- 单元测试：各智能体节点独立测试（使用 mock LLM）
- 集成测试：完整工作流测试（使用真实 LLM，标记为 slow）
- 覆盖率 > 80%

### 8. 文档
- README.md：安装、配置、使用、架构说明
- API 文档：各智能体输入输出格式
- 示例：至少2个使用示例

### 9. 依赖
- langgraph >= 0.2.0
- langchain >= 0.2.0
- openai >= 1.30.0
- pyyaml >= 6.0
- pytest >= 7.0.0
- python-dotenv >= 1.0.0

### 10. 代码质量
- 类型提示完整
- 文档字符串完整
- 错误处理健壮
- 日志输出清晰
- 遵循 PEP8
"""

print("="*70)
print("🚀 启动首个项目：LangGraph Multi-Agent 协作系统")
print("="*70)
print(f"\n需求:\n{requirements}\n")
print("="*70)
print("开始运行...")
print("="*70)

final_state = orchestrator.run(requirements)

print("\n\n" + "="*70)
print("✅ 项目开发完成")
print("="*70)
