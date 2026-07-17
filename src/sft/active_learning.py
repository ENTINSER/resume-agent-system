"""SFT 主动学习采样器

基于不确定性和多样性选择最值得训练的样本子集。
"""

import random
from dataclasses import replace
from typing import Dict, Any, List, Optional

import numpy as np

from src.core.logger import logger
from src.sft.collectors import SFTSample


DEFAULT_EMBEDDING_SIZE = 64


def _sample_text(sample: SFTSample) -> str:
    """将样本转换为用于生成伪嵌入的文本。"""
    return f"{sample.role}\n{sample.instruction}\n{sample.input_text}\n{sample.output}"


def _deterministic_embedding(text: str, vector_size: int = DEFAULT_EMBEDDING_SIZE) -> List[float]:
    """基于文本内容生成确定性伪嵌入，避免依赖外部 embedding 服务。"""
    import hashlib

    h = hashlib.md5(text.encode("utf-8")).digest()
    vec = []
    for i in range(vector_size):
        val = (h[i % len(h)] + i) % 200 - 100
        vec.append(val / 100.0)
    return vec


def _get_embeddings(samples: List[SFTSample], vector_size: int = DEFAULT_EMBEDDING_SIZE) -> List[List[float]]:
    """为样本列表生成确定性伪嵌入。"""
    return [_deterministic_embedding(_sample_text(s), vector_size) for s in samples]


class UncertaintySampler:
    """基于质量分数的不确定性采样器。

    quality_score 越低表示模型越不确定，越应该被选中进行训练。
    低于 ``min_score`` 的样本会被视为明显劣质样本而排除。
    """

    def __init__(self, min_score: float = 0.0, score_key: str = "quality_score"):
        self.min_score = min_score
        self.score_key = score_key

    def sample(self, samples: List[SFTSample], k: int) -> List[SFTSample]:
        if not samples or k <= 0:
            return []
        candidates = [
            s for s in samples
            if getattr(s, self.score_key, 0.0) >= self.min_score
        ]
        # 分数越低不确定性越高，优先选择
        candidates.sort(key=lambda s: getattr(s, self.score_key, 0.0))
        return candidates[:k]


class DiversitySampler:
    """基于嵌入聚类的多样性采样器。"""

    def __init__(self, random_fallback: bool = True, random_state: int = 42):
        self.random_fallback = random_fallback
        self.random_state = random_state

    def sample(
        self,
        samples: List[SFTSample],
        embeddings: List[List[float]],
        k: int,
    ) -> List[SFTSample]:
        if not samples or k <= 0:
            return []
        n = len(samples)
        if n != len(embeddings):
            raise ValueError("samples 与 embeddings 长度必须一致")
        k = min(k, n)

        embeddings_arr = np.asarray(embeddings, dtype=float)
        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            labels = kmeans.fit_predict(embeddings_arr)
            selected = []
            for cluster_id in range(k):
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                if not cluster_indices:
                    continue
                centroid = kmeans.cluster_centers_[cluster_id]
                best_idx = min(
                    cluster_indices,
                    key=lambda i: float(np.linalg.norm(embeddings_arr[i] - centroid)),
                )
                selected.append(samples[best_idx])
            return selected
        except Exception as e:
            logger.warning(f"[DiversitySampler] KMeans 失败，回退到随机采样: {e}")
            if self.random_fallback:
                rng = random.Random(self.random_state)
                return rng.sample(samples, k)
            return samples[:k]


class CombinedSampler:
    """结合不确定性与多样性的综合采样器。

    先过滤掉质量分过低的样本，再分别用不确定性采样和多样性采样
    各取一半，合并去重后裁剪到 k 条。
    """

    def __init__(
        self,
        min_score: float = 0.0,
        uncertainty_ratio: float = 0.5,
        random_state: int = 42,
    ):
        self.min_score = min_score
        self.uncertainty_ratio = uncertainty_ratio
        self.random_state = random_state
        self.uncertainty_sampler = UncertaintySampler(min_score=min_score)
        self.diversity_sampler = DiversitySampler(random_state=random_state)

    def sample(
        self,
        samples: List[SFTSample],
        embeddings: Optional[List[List[float]]] = None,
        k: int = 100,
    ) -> List[SFTSample]:
        if not samples or k <= 0:
            return []

        filtered = [s for s in samples if s.quality_score >= self.min_score]
        if not filtered:
            return []

        k = min(k, len(filtered))
        uncertainty_k = max(1, int(k * self.uncertainty_ratio))
        diversity_k = max(1, k - uncertainty_k)

        selected: List[SFTSample] = []
        selected.extend(self.uncertainty_sampler.sample(filtered, uncertainty_k))

        if embeddings is None:
            filtered_embeddings = _get_embeddings(filtered)
        else:
            if len(embeddings) != len(samples):
                raise ValueError("samples 与 embeddings 长度必须一致")
            sample_index = {id(s): i for i, s in enumerate(samples)}
            filtered_embeddings = [
                embeddings[sample_index[id(s)]] for s in filtered
            ]
        selected.extend(
            self.diversity_sampler.sample(filtered, filtered_embeddings, diversity_k)
        )

        # 按插入顺序去重
        seen = set()
        unique = []
        for s in selected:
            key = (s.role, s.instruction, s.input_text, s.output)
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique[:k]


def sample_for_role(samples: List[SFTSample], state: Dict[str, Any], k: int) -> List[SFTSample]:
    """为指定角色从候选样本中采样子集。

    若样本自身 ``quality_score`` 为 0，则回退到 ``state.evaluation_result.overall_score``。
    多样性通过基于样本文本的确定性伪嵌入实现，无需外部模型。
    """
    if not samples or k <= 0:
        return []

    evaluation = state.get("evaluation_result", {}) or {}
    fallback_score = float(evaluation.get("overall_score", 0.0))

    enriched = []
    for s in samples:
        score = s.quality_score if s.quality_score else fallback_score
        enriched.append(replace(s, quality_score=score))

    embeddings = _get_embeddings(enriched)
    sampler = CombinedSampler()
    return sampler.sample(enriched, embeddings=embeddings, k=k)
