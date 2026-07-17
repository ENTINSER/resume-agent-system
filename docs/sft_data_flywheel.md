# SFT 数据飞轮

## 1. 为什么需要 SFT 数据飞轮

平台每天运行大量 D-B-E-C 任务，产生高质量轨迹：

```
需求 → 项目架构 → 代码 → 测试结果 → 评估报告
```

这些轨迹是训练专属 LLM 的宝贵数据。通过 SFT 数据飞轮，我们可以：
- 让 D 模型更擅长架构设计
- 让 B 模型更擅长代码生成
- 让 E 模型更擅长质量评估
- 让 C 模型更擅长训练脚本生成

最终形成"越用越强"的闭环。

## 2. 架构

```
┌─────────────┐     成功任务        ┌─────────────────┐
│  D/B/E/C    │ ─────────────────► │ SFTCollector    │
│  智能体      │                    │ Pipeline        │
└─────────────┘                    └────────┬────────┘
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
              DCollector              BCollector              ECollector
         (需求→架构)             (需求+架构→代码)        (代码+测试→评估)
                    │                       │                       │
                    └───────────────────────┼───────────────────────┘
                                            ▼
                                     QualityGate
                                (评分/测试/安全/去重)
                                            │
                    ┌───────────────────────┼───────────────────────┐
                    ▼                       ▼                       ▼
              SFTDataset              SFTDataset              SFTDataset
                  (D)                    (B)                     (E)
                    │                       │                       │
                    └───────────────────────┼───────────────────────┘
                                            ▼
                                       SFTTrainer
                                  (MLX LoRA / 云端预留)
                                            │
                                            ▼
                                       Model Registry
```

## 3. 多角色数据集

| 角色 | 输入 | 输出 | 用途 |
|------|------|------|------|
| D | 需求 | 项目架构 JSON | 训练架构师模型 |
| B | 需求 + 架构 | 完整项目代码 | 训练开发模型 |
| E | 代码 + 测试 + 静态指标 | 评估报告 JSON | 训练评估模型 |
| C | 代码片段 | 函数实现 | 训练代码补全模型 |

## 4. 数据质量门

只有满足以下条件的样本才会进入数据集：

1. `evaluation_result.passed == true`
2. `evaluation_result.overall_score >= SFT_MIN_SCORE`（默认 70）
3. 无语法错误
4. 测试全部通过（可选）
5. 不含 secrets（API Key、密码、Token 等）
6. 不含 PII（手机号、邮箱、身份证号等）
7. 不与已有样本重复

## 5. 格式转换

数据集默认以内部格式存储，训练前可导出为：

- **Alpaca**：`{instruction, input, output}`
- **ShareGPT / OpenAI**：`{messages: [...]}`
- **MLX**：`{text: prompt + completion}`

## 6. 训练触发

### 自动触发

任务完成后：
1. `agent_task.py` 调用 `SFTFlywheel.process(state)`
2. 高质量样本写入各角色数据集
3. 若某角色样本数 >= `SFT_MIN_SAMPLES`（默认 50），触发 MLX LoRA 训练

### 手动触发

```bash
source venv/bin/activate
python -m src.sft.train --role B --min-samples 50
```

## 7. 配置

环境变量：

```bash
SFT_ENABLED=true         # 启用
SFT_MIN_SCORE=70         # 纳入数据集的最低评分
SFT_MIN_SAMPLES=50       # 自动触发训练的最小样本数
SFT_ROLES=D,B,E,C        # 启用角色
```

`config/config.yaml`：

```yaml
sft:
  enabled: true
  min_score: 70
  min_samples: 50
  roles: [D, B, E, C]
  output_formats: [alpaca, sharegpt, mlx]
```

## 8. 安全与治理

- 样本中不保存完整源码路径和敏感信息
- 自动扫描 secrets 和 PII
- 每条样本记录来源 `task_id`，支持血缘追溯
- 数据集版本快照，支持回滚
- 模型文件权限限制为 owner 可读写

## 9. 企业级扩展路径

| 当前实现 | 扩展方向 |
|----------|----------|
| 本地 JSONL 存储 | 对象存储 + Delta Lake / Hive |
| 本地模型仓库 | MLflow Model Registry |
| MLX 本地训练 | 阿里云 PAI / AWS SageMaker / 火山引擎 |
| 简单 PII 正则 | 企业级 DLP / Presidio |
| 文件系统版本 | DVC / Git LFS |

## 10. 与 C 智能体的关系

C 智能体现在使用 `src/sft/trainer.py` 生成训练脚本：

```python
from src.sft.collectors import CCollector
from src.sft.trainer import SFTTrainer

samples = CCollector().collect(state)
trainer = SFTTrainer(base_model="...")
artifacts = trainer.prepare_training_artifacts(...)
```

当用户需求涉及训练/微调时，C 智能体会自动生成 MLX LoRA 训练脚本。
