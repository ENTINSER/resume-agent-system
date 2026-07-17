"""SFT 格式转换器单元测试"""

import json
import tempfile

from src.sft.formats import FormatConverter


class TestFormatConverter:
    def test_to_alpaca(self):
        sample = {"instruction": "写代码", "input": "", "output": "print('hi')"}
        result = FormatConverter.to_alpaca(sample)
        assert result["instruction"] == "写代码"
        assert result["output"] == "print('hi')"

    def test_to_sharegpt(self):
        sample = {"role": "B", "instruction": "写代码", "input": "", "output": "print('hi')"}
        result = FormatConverter.to_sharegpt(sample)
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][2]["role"] == "assistant"

    def test_to_mlx(self):
        sample = {"instruction": "写代码", "input": "", "output": "print('hi')"}
        result = FormatConverter.to_mlx(sample)
        assert "<|user|>" in result["text"]
        assert "<|assistant|>" in result["text"]

    def test_convert_dataset(self):
        samples = [
            {"instruction": f"写代码{i}", "input": "", "output": f"print({i})"}
            for i in range(3)
        ]
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        FormatConverter.convert_dataset(samples, path, fmt="alpaca")
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 3
        assert lines[0]["instruction"] == "写代码0"
