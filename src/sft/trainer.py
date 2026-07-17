"""SFT 训练流水线

封装 MLX LoRA 训练脚本生成与执行，预留云端训练接口。
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.logger import logger


DEFAULT_BASE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct"
DEFAULT_LORA_RANK = 8
DEFAULT_LORA_ALPHA = 16
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 3


class SFTTrainer:
    """SFT 训练器"""

    def __init__(
        self,
        base_model: str = DEFAULT_BASE_MODEL,
        lora_rank: int = DEFAULT_LORA_RANK,
        lora_alpha: int = DEFAULT_LORA_ALPHA,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        epochs: int = DEFAULT_EPOCHS,
    ):
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def generate_mlx_script(
        self,
        train_data: str,
        val_data: Optional[str],
        output_dir: str,
    ) -> str:
        """生成 MLX LoRA 训练脚本"""
        script = f'''#!/usr/bin/env python3
"""MLX LoRA 训练脚本 - 自动生成

用法:
    python {output_dir}/train.py

依赖:
    pip install mlx mlx-lm
"""

import mlx.core as mx
from mlx_lm import load, train

MODEL = "{self.base_model}"
TRAIN_DATA = "{train_data}"
VAL_DATA = "{val_data or ''}"
OUTPUT_DIR = "{output_dir}"

LORA_CONFIG = {{
    "rank": {self.lora_rank},
    "alpha": {self.lora_alpha},
    "dropout": 0.05,
    "scale": 10.0,
}}

TRAINING_CONFIG = {{
    "learning_rate": {self.learning_rate},
    "batch_size": {self.batch_size},
    "epochs": {self.epochs},
    "steps_per_eval": 10,
    "adapter_path": f"{{OUTPUT_DIR}}/adapters",
}}

def main():
    print("=" * 60)
    print("MLX LoRA 训练")
    print("=" * 60)
    print(f"模型: {{MODEL}}")
    print(f"训练数据: {{TRAIN_DATA}}")
    print(f"LoRA rank: {self.lora_rank}, alpha: {self.lora_alpha}")
    print(f"学习率: {self.learning_rate}, batch_size: {self.batch_size}, epochs: {self.epochs}")
    print("=" * 60)

    model, tokenizer = load(MODEL)
    train(
        model=model,
        tokenizer=tokenizer,
        path_or_data=TRAIN_DATA,
        valid_data=VAL_DATA if VAL_DATA else None,
        adapter_file=TRAINING_CONFIG["adapter_path"],
        lora_parameters=LORA_CONFIG,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        batch_size=TRAINING_CONFIG["batch_size"],
        iters=TRAINING_CONFIG["epochs"] * 100,
        steps_per_eval=TRAINING_CONFIG["steps_per_eval"],
    )

    print("训练完成!")
    print(f"适配器已保存到: {{TRAINING_CONFIG['adapter_path']}}")

if __name__ == "__main__":
    main()
'''
        return script

    def prepare_training_artifacts(
        self,
        role: str,
        dataset_path: str,
        val_path: Optional[str],
        output_dir: str,
    ) -> Dict[str, Any]:
        """准备训练脚本与 README"""
        os.makedirs(output_dir, exist_ok=True)

        script = self.generate_mlx_script(dataset_path, val_path, output_dir)
        script_path = os.path.join(output_dir, "train.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        readme = f"""# SFT 训练任务 - {role}

## 配置
- 基础模型: {self.base_model}
- LoRA rank: {self.lora_rank}
- LoRA alpha: {self.lora_alpha}
- 学习率: {self.learning_rate}
- Batch size: {self.batch_size}
- Epochs: {self.epochs}

## 数据
- 训练数据: {dataset_path}
- 验证数据: {val_path or '无'}

## 运行训练

```bash
pip install mlx mlx-lm
python {script_path}
```

## 说明
训练完成后，LoRA 适配器将保存在 `{output_dir}/adapters`。
"""
        readme_path = os.path.join(output_dir, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme)

        return {
            "role": role,
            "base_model": self.base_model,
            "script_path": script_path,
            "readme_path": readme_path,
            "output_dir": output_dir,
            "train_data": dataset_path,
            "val_data": val_path,
        }

    def train(self, role: str, dataset, output_base_dir: str = "models/sft") -> Dict[str, Any]:
        """为指定角色准备 SFT 训练"""
        split = dataset.split(test_size=0.1)
        role_dir = Path(output_base_dir) / role
        role_dir.mkdir(parents=True, exist_ok=True)

        # 导出 MLX 格式
        train_path = str(role_dir / "train_mlx.jsonl")
        val_path = str(role_dir / "val_mlx.jsonl")

        from src.sft.formats import FormatConverter
        FormatConverter.convert_dataset(split["train"], train_path, fmt="mlx")
        if split["val"]:
            FormatConverter.convert_dataset(split["val"], val_path, fmt="mlx")
        else:
            val_path = None

        output_dir = str(role_dir / "lora_output")
        artifacts = self.prepare_training_artifacts(role, train_path, val_path, output_dir)

        logger.info(f"[SFT Trainer] {role} 训练产物已准备: {output_dir}")
        return artifacts
