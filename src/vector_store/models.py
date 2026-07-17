"""向量文档模型"""

from dataclasses import dataclass, field
from typing import Optional, Set


# 内置支持的文档类型；业务代码可扩展，但保留核心类型校验
SUPPORTED_DOC_TYPES: Set[str] = {
    "requirement",
    "code",
    "evaluation",
    "project_summary",
    # Phase 5 新增
    "external_code",
    "external_doc",
    "historical_code",
    "historical_doc",
    "synthetic_example",
    "test_failure",
    "test_success",
}


@dataclass
class VectorDocument:
    """存入向量数据库的文档"""

    id: str
    doc_type: str
    session_id: str
    task_id: Optional[str]
    content: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[list] = None

    def __post_init__(self):
        if self.doc_type not in SUPPORTED_DOC_TYPES:
            raise ValueError(f"不支持的 doc_type: {self.doc_type}")
