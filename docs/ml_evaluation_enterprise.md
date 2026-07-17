# 企业级 ML 自动化评估

## 1. 为什么需要 ML 评估

LLM 评估虽然能力强，但存在以下问题：
- **成本高**：每次调用都消耗 token
- **不稳定**：模型版本、temperature、prompt 变化会导致评分波动
- **延迟大**：需要等待 LLM 响应
- **难以 A/B 测试**：缺乏统一基线

ML 自动化评估基于历史数据训练轻量模型，提供：
- **低成本**：本地 CPU 推理，毫秒级响应
- **一致性**：基于结构化特征，评分稳定
- **可解释性**：特征重要性展示评分依据
- **可迭代**：数据越多，模型越准

## 2. 架构

```
┌──────────────┐     评估结果      ┌─────────────────┐
│  E 智能体     │ ───────────────► │  EvaluationDataset │
│ (LLM + ML)   │                  │  (JSONL 样本仓库)   │
└──────────────┘                  └────────┬────────┘
       ▲                                   │
       │ 预测分数                           │ 训练
       │                                   ▼
┌──────┴───────┐                  ┌─────────────────┐
│ MLEvaluation │ ◄─────────────── │  ModelTrainer   │
│   Service    │   production 模型 │  + ModelRegistry │
└──────────────┘                  └─────────────────┘
       │
       ▼
┌─────────────────┐
│  ModelMonitor   │
│  漂移检测/指标   │
└─────────────────┘
```

## 3. 数据飞轮

1. 每次任务完成，E 智能体产出 `evaluation_result`
2. `EvaluationDataset.append(state, result)` 提取特征并写入样本
3. 达到 `min_samples`（默认 20）后，Celery 自动触发 `train_ml_evaluator`
4. `ModelTrainer` 训练回归 + 分类模型
5. `ModelRegistry` 注册新模型，默认 stage 为 `shadow`
6. Shadow 模式运行一段时间后，验证效果可 promote 到 `production`

## 4. 评估服务模式

### 4.1 Shadow 模式（默认）

仅记录 ML 预测分数，不改变最终决策。用于：
- 收集 ML 与 LLM 的对比数据
- 验证模型稳定性
- 无风险上线新模型

### 4.2 Online 模式

ML 分数按权重融入最终评分：
```
final_score = (1 - weight_ml) * llm_score + weight_ml * ml_score
```

默认 `weight_ml = 0.3`。适合 ML 模型已经过充分验证的场景。

### 4.3 降级能力

当 LLM 不可用时，若存在 production 模型，可直接使用 ML 评分作为最终评分，保证服务可用性。

## 5. 配置

环境变量：

```bash
ML_EVAL_ENABLED=true          # 是否启用
ML_EVAL_MODE=shadow           # shadow / online / disabled
ML_EVAL_WEIGHT_ML=0.3         # Online 模式下 ML 权重
ML_EVAL_MIN_SAMPLES=20        # 自动训练最小样本数
```

`config/config.yaml`：

```yaml
ml_eval:
  enabled: true
  mode: shadow
  weight_ml: 0.3
  min_samples: 20
  dataset_version: v1
  model_stage: production
```

## 6. 手动训练

```bash
source venv/bin/activate
python -m src.ml.train --min-samples 20 --stage production
```

训练完成后，模型将注册到 `data/ml_model_registry.json`。

## 7. 模型管理

```python
from src.ml.registry import ModelRegistry, STAGE_PRODUCTION, STAGE_SHADOW

registry = ModelRegistry()
registry.promote("ml_eval_20260630_150712_xxxxxx", STAGE_PRODUCTION)
```

## 8. 监控指标

`ModelMonitor` 输出 Prometheus 格式指标：

```
ml_eval_predictions_total 100
ml_eval_score_average 78.5000
```

可通过 `/metrics` 接口（后续接入 Go Gateway）暴露给 Prometheus。

## 9. 企业级扩展路径

| 当前实现 | 扩展方向 |
|----------|----------|
| 本地 JSONL 样本仓库 | 迁移到对象存储 + Delta Lake / Hive |
| 本地 JSON 模型仓库 | 替换为 MLflow Model Registry |
| 文件系统模型加载 | 替换为模型服务（Triton / TorchServe / 自研） |
| 简单漂移检测 | 接入 Evidently / WhyLogs 做 PSI、KS 检验 |
| Celery 定时训练 | 接入 Airflow / Kubeflow Pipelines |
| Prometheus 文本指标 | 通过 Grafana 可视化 |

## 10. 安全与治理

- 样本中不保存完整代码，仅保留结构化特征与项目 ID
- 模型文件权限限制为 owner 可读写
- 每次模型注册、升级记录审计日志
- 支持 PII 过滤（后续扩展）
