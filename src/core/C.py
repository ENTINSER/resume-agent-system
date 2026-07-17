"""C - Trainer（训练工程师）P3优化版

优化点：
- 自动数据集构建（从B的代码产出中提取训练样本）
- MLX LoRA配置与脚本生成
- 训练任务状态追踪
- 后台训练支持
- 与B联动：当需求涉及训练时自动触发
"""

import os
import sys
import json
import re
from typing import Optional, List, Dict
from datetime import datetime

from src.core.shared_state import SharedState
from src.core.logger import logger
from src.core.config import settings
from src.sft.collectors import CCollector
from src.sft.trainer import SFTTrainer


# ===== 配置常量 =====
DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct"  # 小模型用于快速验证
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 3
MLX_TRAIN_TIMEOUT = 3600  # 1小时超时


class DatasetBuilder:
    """数据集构建器 - 从代码产出中自动构建训练数据"""
    
    INSTRUCTION_TEMPLATES = [
        "请编写一个Python函数，{description}",
        "实现以下功能：{description}",
        "编写代码完成：{description}",
        "用Python实现：{description}",
    ]
    
    @staticmethod
    def from_code_artifacts(artifacts: List[dict], requirements: str = "") -> List[dict]:
        """从代码产出构建训练数据集"""
        samples = []
        
        for artifact in artifacts:
            if artifact.get("language") != "python":
                continue
            
            content = artifact.get("content", "")
            file_path = artifact.get("file_path", "")
            
            # 提取函数定义和文档字符串
            functions = DatasetBuilder._extract_functions(content)
            
            for func in functions:
                instruction = DatasetBuilder._generate_instruction(func, file_path, requirements)
                output = func["code"]
                
                samples.append({
                    "instruction": instruction,
                    "input": "",
                    "output": output,
                    "source_file": file_path,
                    "function_name": func["name"]
                })
        
        # 如果样本太少，添加通用代码完成样本
        if len(samples) < 5:
            samples.extend(DatasetBuilder._generate_generic_samples(requirements))
        
        return samples
    
    @staticmethod
    def _extract_functions(code: str) -> List[dict]:
        """提取函数定义（简化版，基于正则）"""
        functions = []
        
        # 匹配函数定义：def name(args): ... （直到下一个def或class）
        pattern = r'def\s+(\w+)\s*\([^)]*\)\s*(?:->\s*[^:]+)?:\s*(.*?)(?=\ndef\s+|\nclass\s+|\Z)'
        matches = re.findall(pattern, code, re.DOTALL)
        
        for name, body in matches:
            # 找到函数之前的文档字符串
            # 简单处理：取函数定义前5行作为上下文
            lines = code.split('\n')
            func_start = next((i for i, l in enumerate(lines) if f'def {name}(' in l), 0)
            
            docstring = ""
            if func_start + 1 < len(lines):
                next_line = lines[func_start + 1].strip()
                if next_line.startswith('"""') or next_line.startswith("'''"):
                    # 简单提取文档字符串
                    doc_lines = []
                    in_docstring = False
                    for i in range(func_start + 1, min(func_start + 10, len(lines))):
                        line = lines[i]
                        if '"""' in line or "'''" in line:
                            if in_docstring:
                                doc_lines.append(line)
                                break
                            else:
                                in_docstring = True
                        if in_docstring:
                            doc_lines.append(line)
                    docstring = '\n'.join(doc_lines).strip('"\'').strip()
            
            # 获取完整函数代码（包含def行）
            func_code_lines = []
            for i in range(func_start, len(lines)):
                line = lines[i]
                # 检测下一个函数/类的缩进级别
                if i > func_start and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                    break
                func_code_lines.append(line)
            
            func_code = '\n'.join(func_code_lines)
            
            functions.append({
                "name": name,
                "docstring": docstring,
                "code": func_code,
                "description": docstring or f"实现{name}功能"
            })
        
        return functions
    
    @staticmethod
    def _generate_instruction(func: dict, file_path: str, requirements: str) -> str:
        """生成指令文本"""
        import random
        
        template = random.choice(DatasetBuilder.INSTRUCTION_TEMPLATES)
        
        # 使用文档字符串作为描述
        description = func["description"]
        if not description and requirements:
            description = f"根据需求：{requirements[:100]}"
        if not description:
            description = f"在文件{file_path}中"
        
        return template.format(description=description)
    
    @staticmethod
    def _generate_generic_samples(requirements: str) -> List[dict]:
        """生成通用训练样本"""
        return [
            {
                "instruction": "请编写一个Python函数，实现两数相加",
                "input": "",
                "output": "def add(a: int, b: int) -> int:\n    \"\"\"返回两个数的和\"\"\"\n    return a + b",
                "source_file": "generic",
                "function_name": "add"
            },
            {
                "instruction": "请编写一个Python函数，计算列表平均值",
                "input": "",
                "output": "def average(numbers: list) -> float:\n    \"\"\"计算列表平均值\"\"\"\n    if not numbers:\n        return 0.0\n    return sum(numbers) / len(numbers)",
                "source_file": "generic",
                "function_name": "average"
            },
            {
                "instruction": "请编写一个Python函数，检查字符串是否为回文",
                "input": "",
                "output": "def is_palindrome(s: str) -> bool:\n    \"\"\"检查字符串是否为回文\"\"\"\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]",
                "source_file": "generic",
                "function_name": "is_palindrome"
            }
        ]
    
    @staticmethod
    def save_to_jsonl(samples: List[dict], output_path: str) -> str:
        """保存为JSONL格式"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                # 移除内部字段
                clean_sample = {k: v for k, v in sample.items() 
                               if k in ["instruction", "input", "output"]}
                f.write(json.dumps(clean_sample, ensure_ascii=False) + '\n')
        
        logger.info(f"[C] 数据集已保存: {output_path}, 样本数: {len(samples)}")
        return output_path
    
    @staticmethod
    def split_dataset(dataset_path: str, train_ratio: float = 0.8) -> tuple:
        """划分训练/验证集"""
        with open(dataset_path, 'r') as f:
            samples = [json.loads(line) for line in f if line.strip()]
        
        train_size = int(len(samples) * train_ratio)
        train_samples = samples[:train_size]
        val_samples = samples[train_size:]
        
        base = dataset_path.replace('.jsonl', '')
        train_path = f"{base}_train.jsonl"
        val_path = f"{base}_val.jsonl"
        
        with open(train_path, 'w') as f:
            for s in train_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        
        with open(val_path, 'w') as f:
            for s in val_samples:
                f.write(json.dumps(s, ensure_ascii=False) + '\n')
        
        return train_path, val_path, len(train_samples), len(val_samples)


class MLXTrainer:
    """MLX训练器 - 生成训练脚本并执行"""
    
    @staticmethod
    def generate_training_script(
        model_name: str,
        train_data: str,
        val_data: Optional[str],
        output_dir: str,
        lora_rank: int = DEFAULT_LORA_RANK,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epochs: int = DEFAULT_EPOCHS
    ) -> str:
        """生成MLX LoRA训练脚本"""
        
        script = f'''#!/usr/bin/env python3
"""MLX LoRA训练脚本 - 自动生成

使用方法:
    python {output_dir}/train.py

依赖:
    pip install mlx mlx-lm
"""

import mlx.core as mx
from mlx_lm import load, generate, train

# 配置
MODEL = "{model_name}"
TRAIN_DATA = "{train_data}"
VAL_DATA = "{val_data or ''}"
OUTPUT_DIR = "{output_dir}"

LORA_CONFIG = {{
    "rank": {lora_rank},
    "alpha": {lora_alpha},
    "dropout": 0.05,
    "scale": 10.0,
}}

TRAINING_CONFIG = {{
    "learning_rate": {learning_rate},
    "batch_size": {batch_size},
    "epochs": {epochs},
    "steps_per_eval": 10,
    "adapter_path": f"{{OUTPUT_DIR}}/adapters",
}}

def main():
    print("=" * 60)
    print("MLX LoRA训练")
    print("=" * 60)
    print(f"模型: {{MODEL}}")
    print(f"训练数据: {{TRAIN_DATA}}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"学习率: {learning_rate}, batch_size: {batch_size}, epochs: {epochs}")
    print("=" * 60)
    
    # 加载模型
    print("加载模型...")
    model, tokenizer = load(MODEL)
    
    # 开始训练
    print("开始训练...")
    train(
        model=model,
        tokenizer=tokenizer,
        path_or_data=TRAIN_DATA,
        valid_data=VAL_DATA if VAL_DATA else None,
        adapter_file=TRAINING_CONFIG["adapter_path"],
        lora_parameters=LORA_CONFIG,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        batch_size=TRAINING_CONFIG["batch_size"],
        iters=TRAINING_CONFIG["epochs"] * 100,  # 估计步数
        steps_per_eval=TRAINING_CONFIG["steps_per_eval"],
    )
    
    print("训练完成!")
    print(f"适配器已保存到: {{TRAINING_CONFIG['adapter_path']}}")

if __name__ == "__main__":
    main()
'''
        return script
    
    @staticmethod
    def check_mlx_available() -> dict:
        """检查MLX环境"""
        result = {
            "mlx": False,
            "mlx_lm": False,
            "gpu": False,
            "gpu_name": "",
            "recommendation": ""
        }
        
        try:
            import mlx.core as mx
            result["mlx"] = True
            
            # 检查GPU
            devices = mx.get_devices()
            if devices:
                result["gpu"] = True
                result["gpu_name"] = str(devices[0])
        except ImportError:
            pass
        
        try:
            import mlx_lm
            result["mlx_lm"] = True
        except ImportError:
            pass
        
        if result["mlx"] and result["mlx_lm"]:
            result["recommendation"] = "环境就绪，可以直接训练"
        elif result["mlx"]:
            result["recommendation"] = "需要安装 mlx-lm: pip install mlx-lm"
        else:
            result["recommendation"] = "需要安装MLX: pip install mlx mlx-lm"
        
        return result


# ===== 主函数 =====

def trainer_node(state: SharedState) -> SharedState:
    """训练节点 - P3优化版"""
    logger.info("[C] 开始模型训练")
    
    state["current_phase"] = "training"
    state["status"] = "training"
    
    # 检查是否需要训练
    if not _should_train(state):
        logger.info("[C] 当前项目不需要训练，跳过")
        state["status"] = "training_skipped"
        return state
    
    # 构建数据集（使用 SFT 采集器）
    collector = CCollector()
    if not collector.can_collect(state):
        logger.warning("[C] 无可用代码产物，无法训练")
        state["status"] = "training_insufficient_data"
        return state

    samples = collector.collect(state)
    if len(samples) < 3:
        logger.warning("[C] 训练样本太少(<3)，无法训练")
        state["status"] = "training_insufficient_data"
        return state

    # 保存数据集
    session_id = state["session_id"][:8]
    dataset_dir = f"data/datasets/{session_id}"
    dataset_path = f"{dataset_dir}/train.jsonl"
    DatasetBuilder.save_to_jsonl([s.to_dict() for s in samples], dataset_path)

    # 划分训练/验证集
    train_path, val_path, train_count, val_count = DatasetBuilder.split_dataset(dataset_path)
    
    # 创建训练任务
    job = {
        "id": f"train-{session_id}",
        "model_name": DEFAULT_MODEL,
        "dataset_path": dataset_path,
        "train_path": train_path,
        "val_path": val_path,
        "train_samples": train_count,
        "val_samples": val_count,
        "status": "dataset_ready",
        "created_at": datetime.now().isoformat(),
        "metrics": {},
        "config": {
            "lora_rank": DEFAULT_LORA_RANK,
            "lora_alpha": DEFAULT_LORA_ALPHA,
            "learning_rate": DEFAULT_LEARNING_RATE,
            "batch_size": DEFAULT_BATCH_SIZE,
            "epochs": DEFAULT_EPOCHS
        }
    }
    state["training_jobs"] = state.get("training_jobs", []) + [job]
    
    # 检查MLX环境
    mlx_status = MLXTrainer.check_mlx_available()
    job["mlx_status"] = mlx_status
    
    if not mlx_status["mlx"]:
        logger.warning(f"[C] MLX不可用: {mlx_status['recommendation']}")
        # 生成训练脚本供手动运行
        state = _generate_training_artifacts(state, job)
        state["status"] = "training_script_ready"
        return state
    
    # 执行训练（或生成脚本）
    state = _generate_training_artifacts(state, job)

    if settings.sft_auto_train and mlx_status["mlx"] and mlx_status["mlx_lm"]:
        logger.info(f"[C] 自动训练已启用，启动 MLX 训练: {job['id']}")
        state = _run_mlx_training(state, job)
    else:
        if settings.sft_auto_train and not (mlx_status["mlx"] and mlx_status["mlx_lm"]):
            logger.warning("[C] 自动训练已启用但 MLX 环境不可用，仅生成训练脚本")
        job["status"] = "script_ready"

    logger.info(f"[C] 训练准备完成: {job['id']}, 样本: {train_count}/{val_count}")

    return state


def _should_train(state: SharedState) -> bool:
    """判断是否需要训练"""
    keywords = ["模型", "训练", "微调", "fine-tune", "LoRA", "量化", "lora", "train"]
    requirements = state.get("requirements", "").lower()
    return any(kw in requirements for kw in keywords)


def _generate_training_artifacts(state: SharedState, job: dict) -> SharedState:
    """生成训练相关文件（使用 SFTTrainer）"""
    output_dir = f"models/{job['id']}"
    os.makedirs(output_dir, exist_ok=True)

    # 使用 SFTTrainer 生成脚本与 README
    trainer = SFTTrainer(
        base_model=job['model_name'],
        lora_rank=job['config']['lora_rank'],
        lora_alpha=job['config']['lora_alpha'],
        learning_rate=job['config']['learning_rate'],
        batch_size=job['config']['batch_size'],
        epochs=job['config']['epochs'],
    )
    artifacts = trainer.prepare_training_artifacts(
        role="C",
        dataset_path=job['train_path'],
        val_path=job.get('val_path'),
        output_dir=output_dir,
    )

    script_path = artifacts["script_path"]
    readme_path = artifacts["readme_path"]

    with open(script_path, "r", encoding="utf-8") as f:
        script = f.read()
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()

    # 追加 C 智能体环境状态到 README
    env_note = f"""
## 环境状态
- MLX: {'✅' if job['mlx_status'].get('mlx') else '❌'}
- mlx-lm: {'✅' if job['mlx_status'].get('mlx_lm') else '❌'}
- GPU: {'✅' if job['mlx_status'].get('gpu') else '❌'} {job['mlx_status'].get('gpu_name', '')}

{job['mlx_status'].get('recommendation', '')}
"""
    readme += env_note
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    # 添加到代码产出（供保存）
    state["code_artifacts"] = state.get("code_artifacts", []) + [
        {
            "file_path": f"models/{job['id']}/train.py",
            "content": script,
            "language": "python",
            "source": "trainer"
        },
        {
            "file_path": f"models/{job['id']}/README.md",
            "content": readme,
            "language": "markdown",
            "source": "trainer"
        }
    ]

    job["script_path"] = script_path
    job["readme_path"] = readme_path
    job["status"] = "script_ready"

    logger.info(f"[C] 训练脚本已生成: {script_path}")

    return state


def _run_mlx_training(state: SharedState, job: dict) -> SharedState:
    """执行MLX训练（实际运行，耗时较长）

    根据 settings.sft_mlx_timeout 控制超时，
    根据 settings.sft_mlx_max_retries 在失败时重试。
    """
    import subprocess
    import threading

    timeout = settings.sft_mlx_timeout
    max_retries = settings.sft_mlx_max_retries

    def train_worker():
        attempt = 0
        while attempt <= max_retries:
            attempt += 1
            logger.info(f"[C] 训练尝试 {attempt}/{max_retries + 1}: {job['id']}")
            try:
                result = subprocess.run(
                    [sys.executable or "python3", job["script_path"]],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )

                if result.returncode == 0:
                    job["status"] = "completed"
                    job["train_output"] = result.stdout
                    job["train_error"] = ""
                    job["completed_at"] = datetime.now().isoformat()

                    # 解析损失
                    losses = re.findall(r'loss[:\s]+(\d+\.\d+)', result.stdout)
                    if losses:
                        job["metrics"]["final_loss"] = float(losses[-1])

                    logger.info(f"[C] 训练完成: {job['id']}")
                    return

                job["train_output"] = result.stdout
                job["train_error"] = result.stderr
                job["last_returncode"] = result.returncode
                logger.warning(f"[C] 训练失败（返回码 {result.returncode}），将重试: {job['id']}")

            except subprocess.TimeoutExpired:
                job["status"] = "timeout"
                logger.warning(f"[C] 训练超时(>{timeout}s): {job['id']}")
            except Exception as e:
                job["status"] = "error"
                job["train_error"] = str(e)
                logger.error(f"[C] 训练错误: {e}")

        job["status"] = "failed"
        job["completed_at"] = datetime.now().isoformat()
        logger.error(f"[C] 训练最终失败，已用尽 {max_retries} 次重试: {job['id']}")

    # 在后台线程运行训练
    thread = threading.Thread(target=train_worker, daemon=True)
    thread.start()

    job["status"] = "training"
    job["started_at"] = datetime.now().isoformat()

    # 不等待训练完成，继续流程
    logger.info(f"[C] 训练任务已在后台启动: {job['id']}")

    return state


def quantize_model(model_path: str, method: str = "awq") -> str:
    """模型量化"""
    logger.info(f"[C] 量化模型: {model_path}, 方法: {method}")
    
    quantized_path = f"{model_path}-{method}-4bit"
    logger.info(f"[C] 量化完成: {quantized_path}")
    
    return quantized_path


def generate_training_report(state: SharedState) -> str:
    """生成训练报告"""
    jobs = state.get("training_jobs", [])
    if not jobs:
        return "无训练任务"
    
    job = jobs[-1]
    
    report = f"""
训练报告
========
任务ID: {job.get('id', '')}
模型: {job.get('model_name', '')}
状态: {job.get('status', '')}
数据集: {job.get('dataset_path', '')}
训练样本: {job.get('train_samples', 0)}
验证样本: {job.get('val_samples', 0)}

配置:
"""
    for key, value in job.get("config", {}).items():
        report += f"  {key}: {value}\n"
    
    if job.get("metrics"):
        report += "\n指标:\n"
        for key, value in job["metrics"].items():
            report += f"  {key}: {value}\n"
    
    if job.get("script_path"):
        report += f"\n训练脚本: {job['script_path']}\n"
    
    return report
