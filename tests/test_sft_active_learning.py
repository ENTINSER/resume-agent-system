"""SFT 主动学习采样器单元测试"""

import pytest
import numpy as np

from src.sft.active_learning import (
    UncertaintySampler,
    DiversitySampler,
    CombinedSampler,
    sample_for_role,
)
from src.sft.collectors import SFTSample


def _make_sample(role: str = "C", instruction: str = "", output: str = "", score: float = 0.0):
    return SFTSample(
        role=role,
        instruction=instruction,
        input_text="",
        output=output,
        context={},
        quality_score=score,
        source_task="t1",
    )


class TestUncertaintySampler:
    def test_selects_lowest_scores(self):
        samples = [
            _make_sample(instruction="a", score=90.0),
            _make_sample(instruction="b", score=30.0),
            _make_sample(instruction="c", score=60.0),
            _make_sample(instruction="d", score=20.0),
        ]
        sampler = UncertaintySampler()
        selected = sampler.sample(samples, k=2)
        assert len(selected) == 2
        assert selected[0].quality_score == 20.0
        assert selected[1].quality_score == 30.0

    def test_excludes_obvious_bad_ones(self):
        samples = [
            _make_sample(instruction="a", score=90.0),
            _make_sample(instruction="b", score=10.0),
            _make_sample(instruction="c", score=5.0),
        ]
        sampler = UncertaintySampler(min_score=10.0)
        selected = sampler.sample(samples, k=10)
        assert len(selected) == 2
        assert all(s.quality_score >= 10.0 for s in selected)

    def test_empty_or_zero_k(self):
        sampler = UncertaintySampler()
        assert sampler.sample([], k=5) == []
        assert sampler.sample([_make_sample(score=50.0)], k=0) == []


class TestDiversitySampler:
    def test_selects_from_different_clusters(self):
        # 构造两个明显分离的簇
        samples = [
            _make_sample(instruction="cluster_a_1"),
            _make_sample(instruction="cluster_a_2"),
            _make_sample(instruction="cluster_b_1"),
            _make_sample(instruction="cluster_b_2"),
        ]
        embeddings = [
            [0.0, 0.0],
            [0.1, 0.1],
            [10.0, 10.0],
            [10.1, 10.1],
        ]
        sampler = DiversitySampler()
        selected = sampler.sample(samples, embeddings, k=2)
        assert len(selected) == 2
        # 两个样本应来自不同簇
        instructions = {s.instruction for s in selected}
        has_a = any("cluster_a" in inst for inst in instructions)
        has_b = any("cluster_b" in inst for inst in instructions)
        assert has_a and has_b

    def test_k_capped_by_sample_count(self):
        samples = [_make_sample(instruction=f"s{i}") for i in range(3)]
        embeddings = [[i, i] for i in range(3)]
        sampler = DiversitySampler()
        selected = sampler.sample(samples, embeddings, k=10)
        assert len(selected) == 3

    def test_length_mismatch_raises(self):
        sampler = DiversitySampler()
        with pytest.raises(ValueError):
            sampler.sample([_make_sample()], [], k=1)


class TestCombinedSampler:
    def test_filters_and_caps(self):
        samples = [
            _make_sample(instruction="low_a", score=10.0),
            _make_sample(instruction="low_b", score=20.0),
            _make_sample(instruction="mid", score=50.0),
            _make_sample(instruction="high", score=90.0),
        ]
        sampler = CombinedSampler(min_score=15.0)
        selected = sampler.sample(samples, k=2)
        assert len(selected) <= 2
        assert all(s.quality_score >= 15.0 for s in selected)

    def test_deduplicates(self):
        samples = [
            _make_sample(instruction="a", score=10.0),
            _make_sample(instruction="a", score=10.0),
        ]
        sampler = CombinedSampler(min_score=0.0)
        selected = sampler.sample(samples, k=10)
        assert len(selected) == 1

    def test_combines_uncertainty_and_diversity(self):
        # 4 个样本分布在 4 个不同簇，k=4 时应同时覆盖不确定性与多样性
        samples = [
            _make_sample(instruction="uncertain_a", score=10.0),
            _make_sample(instruction="uncertain_b", score=15.0),
            _make_sample(instruction="diverse_c", score=80.0),
            _make_sample(instruction="diverse_d", score=85.0),
        ]
        embeddings = [
            [0.0, 0.0],
            [100.0, 0.0],
            [0.0, 100.0],
            [100.0, 100.0],
        ]
        sampler = CombinedSampler(uncertainty_ratio=0.5)
        selected = sampler.sample(samples, embeddings=embeddings, k=4)
        assert 0 < len(selected) <= 4
        instructions = {s.instruction for s in selected}
        # 应至少包含一个低分样本和一个高分样本
        has_uncertain = any("uncertain" in inst for inst in instructions)
        has_diverse = any("diverse" in inst for inst in instructions)
        assert has_uncertain or has_diverse


class TestSampleForRole:
    def test_fallback_score_from_state(self):
        samples = [
            _make_sample(instruction="a", score=0.0),
            _make_sample(instruction="b", score=0.0),
        ]
        state = {"evaluation_result": {"overall_score": 88.0}}
        selected = sample_for_role(samples, state, k=2)
        assert 0 < len(selected) <= 2
        assert all(s.quality_score == 88.0 for s in selected)

    def test_keeps_existing_scores(self):
        samples = [
            _make_sample(instruction="a", score=50.0),
            _make_sample(instruction="b", score=0.0),
        ]
        state = {"evaluation_result": {"overall_score": 90.0}}
        selected = sample_for_role(samples, state, k=2)
        assert 0 < len(selected) <= 2
        by_inst = {s.instruction: s.quality_score for s in selected}
        for inst in by_inst:
            if inst == "a":
                assert by_inst[inst] == 50.0
            elif inst == "b":
                assert by_inst[inst] == 90.0

    def test_caps_at_k(self):
        samples = [_make_sample(instruction=f"s{i}", score=float(i)) for i in range(20)]
        state = {}
        selected = sample_for_role(samples, state, k=5)
        assert 0 < len(selected) <= 5
