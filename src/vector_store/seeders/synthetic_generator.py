"""合成示例种子

不依赖历史项目，用人工编写的高质量示例预先填充 RAG，
帮助新系统在没有历史数据时也能检索到参考模式。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.core.config import settings
from src.core.logger import logger
from src.vector_store.indexer import _chunk_text, _doc_id
from src.vector_store.models import VectorDocument
from src.vector_store.vector_store import get_vector_store


DEFAULT_CATALOG_PATH = Path("data/synthetic_examples/catalog.yaml")


def _default_catalog() -> List[Dict[str, Any]]:
    """内置最小示例，保证即使 catalog 缺失也能注入基础种子"""
    return [
        {
            "id": "example_requirement_decomposition",
            "doc_type": "synthetic_example",
            "title": "需求拆解示例",
            "content": (
                "需求：开发一个支持四则运算的 Python 计算器 CLI。\n"
                "拆解：\n"
                "1. 解析命令行参数（argparse）\n"
                "2. 实现 add/sub/mul/div 四个函数，处理除零\n"
                "3. 编写单元测试覆盖正例、边界、异常\n"
                "4. 输出帮助文档与示例\n"
                "验收标准：pytest 100% 通过，代码圈复杂度 < 5。"
            ),
            "metadata": {"domain": "cli", "tags": ["calculator", "argparse", "pytest"]},
        },
        {
            "id": "example_clean_python_code",
            "doc_type": "synthetic_example",
            "title": "高质量 Python 代码示例",
            "content": (
                "def divide(a: float, b: float) -> float:\n"
                "    if b == 0:\n"
                "        raise ValueError('除数不能为 0')\n"
                "    return a / b\n\n"
                "def test_divide_by_zero():\n"
                "    with pytest.raises(ValueError):\n"
                "        divide(1, 0)\n\n"
                "要点：显式类型标注、边界异常、单一职责、配套测试。"
            ),
            "metadata": {"domain": "python", "tags": ["function", "testing", "typing"]},
        },
        {
            "id": "example_evaluation_rubric",
            "doc_type": "synthetic_example",
            "title": "代码评估维度示例",
            "content": (
                "评估一个生成项目时，从以下维度打分（0-100）：\n"
                "- code_quality: 可读性、命名、结构、类型标注\n"
                "- functionality: 是否满足需求、边界处理\n"
                "- performance: 时间/空间复杂度、资源使用\n"
                "- documentation: README、注释、使用示例\n"
                "overall_score >= 80 且 functionality >= 70 方可通过。"
            ),
            "metadata": {"domain": "evaluation", "tags": ["rubric", "scoring"]},
        },
        {
            "id": "example_test_failure_pattern",
            "doc_type": "synthetic_example",
            "title": "常见测试失败模式",
            "content": (
                "失败模式 1：ModuleNotFoundError 因为未安装依赖。\n"
                "修复：在 requirements.txt 中声明依赖，测试前执行 pip install。\n\n"
                "失败模式 2：断言失败因为浮点精度。\n"
                "修复：使用 pytest.approx 或 math.isclose。\n\n"
                "失败模式 3：未捕获异常导致测试崩溃。\n"
                "修复：在实现中显式校验输入并抛出清晰异常。"
            ),
            "metadata": {"domain": "testing", "tags": ["failure-pattern", "debugging"]},
        },
    ]


class SyntheticExampleGenerator:
    """合成示例生成器"""

    def __init__(
        self,
        catalog_path: Optional[Path] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        self.catalog_path = Path(catalog_path or DEFAULT_CATALOG_PATH)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _load_examples(self) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        if self.catalog_path.exists():
            try:
                with self.catalog_path.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "examples" in data:
                    examples = data["examples"]
                elif isinstance(data, list):
                    examples = data
                logger.info(f"[SyntheticGenerator] 加载 {len(examples)} 条外部合成示例")
            except Exception as e:
                logger.warning(f"[SyntheticGenerator] 读取 catalog 失败，使用默认示例: {e}")
        # 如果外部 catalog 为空或不存在，追加默认示例
        if not examples:
            examples = _default_catalog()
        return examples

    def seed(self) -> Dict[str, int]:
        vector_store = get_vector_store()
        if vector_store is None:
            return {"indexed": 0}

        examples = self._load_examples()
        docs: List[VectorDocument] = []

        for ex in examples:
            ex_id = ex.get("id", "")
            doc_type = ex.get("doc_type", "synthetic_example")
            title = ex.get("title", "")
            content = ex.get("content", "")
            meta = dict(ex.get("metadata") or {})
            meta.update({"title": title, "example_id": ex_id})

            if not content:
                continue

            # 整篇示例索引一条，保证标题/主题召回
            docs.append(VectorDocument(
                id=_doc_id(f"synthetic|{ex_id}|full"),
                doc_type=doc_type,
                session_id="",
                task_id=None,
                content=f"合成示例: {title}\n\n{content}",
                metadata=meta,
            ))

            # 对长示例再切片索引
            for idx, chunk in _chunk_text(content, self.chunk_size, self.chunk_overlap):
                docs.append(VectorDocument(
                    id=_doc_id(f"synthetic|{ex_id}|chunk|{idx}"),
                    doc_type=doc_type,
                    session_id="",
                    task_id=None,
                    content=f"合成示例片段: {title}\n\n{chunk}",
                    metadata={**meta, "chunk_index": idx},
                ))

        if docs:
            vector_store.upsert(docs)

        logger.info(f"[SyntheticGenerator] 合成示例索引完成: {len(docs)} 文档")
        return {"indexed": len(docs), "examples": len(examples)}
