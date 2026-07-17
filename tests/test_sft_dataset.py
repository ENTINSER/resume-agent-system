"""SFT 数据集仓库单元测试"""

import tempfile

import pytest

from src.sft.dataset import SFTDataset
from src.sft.collectors import SFTSample


class TestSFTDataset:
    @pytest.fixture
    def dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SFTDataset(role="B", dataset_dir=tmpdir, version="v1")

    def test_add_and_load(self, dataset):
        samples = [
            SFTSample(role="B", instruction="写代码", input_text="", output="print('hi')", context={}),
        ]
        added = dataset.add_samples(samples)
        assert added == 1
        loaded = dataset.load()
        assert len(loaded) == 1
        assert loaded[0]["role"] == "B"

    def test_deduplicate(self, dataset):
        samples = [
            SFTSample(role="B", instruction="写代码", input_text="", output="print('hi')", context={}),
            SFTSample(role="B", instruction="写代码", input_text="", output="print('hi')", context={}),
        ]
        added = dataset.add_samples(samples)
        assert added == 1

    def test_split(self, dataset):
        samples = [
            SFTSample(role="B", instruction=f"写代码{i}", input_text="", output=f"print({i})", context={})
            for i in range(10)
        ]
        dataset.add_samples(samples)
        split = dataset.split(test_size=0.2)
        assert split["train_count"] == 8
        assert split["val_count"] == 2

    def test_stats(self, dataset):
        samples = [
            SFTSample(role="B", instruction="写代码", input_text="", output="print('hi')", context={}, quality_score=80),
        ]
        dataset.add_samples(samples)
        stats = dataset.stats()
        assert stats["count"] == 1
        assert stats["avg_quality_score"] == 80
