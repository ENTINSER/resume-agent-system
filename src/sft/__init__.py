"""SFT 数据飞轮模块

将高质量任务轨迹转化为监督微调训练数据，支持多角色、多格式、多版本。
"""

from src.sft.collectors import DCollector, BCollector, ECollector, CCollector
from src.sft.quality import QualityGate
from src.sft.dataset import SFTDataset
from src.sft.formats import FormatConverter
from src.sft.registry import SFTRegistry
from src.sft.trainer import SFTTrainer

__all__ = [
    "DCollector",
    "BCollector",
    "ECollector",
    "CCollector",
    "QualityGate",
    "SFTDataset",
    "FormatConverter",
    "SFTRegistry",
    "SFTTrainer",
]
