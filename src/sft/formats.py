"""SFT 格式转换器

支持 Alpaca、ShareGPT / OpenAI、MLX 等格式。
"""

import json
from pathlib import Path
from typing import Dict, Any, List


class FormatConverter:
    """SFT 格式转换"""

    @staticmethod
    def to_alpaca(sample: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "instruction": sample.get("instruction", ""),
            "input": sample.get("input", ""),
            "output": sample.get("output", ""),
        }

    @staticmethod
    def to_sharegpt(sample: Dict[str, Any]) -> Dict[str, Any]:
        """ShareGPT / OpenAI 微调格式"""
        return {
            "messages": [
                {"role": "system", "content": f"你是一个专业的 {sample.get('role', 'AI')} 助手。"},
                {"role": "user", "content": sample.get("instruction", "") + "\n" + sample.get("input", "")},
                {"role": "assistant", "content": sample.get("output", "")},
            ]
        }

    @staticmethod
    def to_mlx(sample: Dict[str, Any]) -> Dict[str, Any]:
        """MLX 训练格式：单条 text"""
        prompt = sample.get("instruction", "") + "\n" + sample.get("input", "")
        completion = sample.get("output", "")
        return {
            "text": f"<|user|>\n{prompt}\n<|assistant|>\n{completion}",
        }

    @staticmethod
    def convert_dataset(
        samples: List[Dict[str, Any]],
        output_path: str,
        fmt: str = "alpaca",
    ) -> str:
        """转换并保存数据集"""
        fmt = fmt.lower()
        converter = {
            "alpaca": FormatConverter.to_alpaca,
            "sharegpt": FormatConverter.to_sharegpt,
            "openai": FormatConverter.to_sharegpt,
            "mlx": FormatConverter.to_mlx,
        }.get(fmt)

        if not converter:
            raise ValueError(f"不支持的格式: {fmt}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for sample in samples:
                converted = converter(sample)
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")

        return output_path
