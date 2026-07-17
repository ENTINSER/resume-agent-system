#!/usr/bin/env python
"""RAG 数据库种子脚本

用法示例：
    venv/bin/python scripts/seed_rag.py --synthetic
    venv/bin/python scripts/seed_rag.py --historical
    venv/bin/python scripts/seed_rag.py --repo https://github.com/example/repo.git --tags python,cli
    venv/bin/python scripts/seed_rag.py --all
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 sys.path，确保可执行脚本无需安装 editable 包也能导入
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.core.config import settings
from src.core.logger import logger
from src.vector_store.seeders import (
    ExternalRepoSeeder,
    HistoricalIndexer,
    SyntheticExampleGenerator,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="为 RAG 向量数据库注入种子数据")
    parser.add_argument("--all", action="store_true", help="执行全部种子任务")
    parser.add_argument("--synthetic", action="store_true", help="注入内置合成示例")
    parser.add_argument("--historical", action="store_true", help="重新索引历史会话与 projects/ 目录")
    parser.add_argument("--repo", type=str, default=None, help="外部仓库本地路径或 git URL")
    parser.add_argument("--ref", type=str, default="main", help="外部仓库分支/标签")
    parser.add_argument("--tags", type=str, default="", help="外部仓库标签，逗号分隔")
    parser.add_argument("--name", type=str, default=None, help="外部仓库别名")
    args = parser.parse_args()

    if not settings.vector_store_enabled:
        logger.error("VECTOR_STORE_ENABLED=false，跳过种子注入")
        return 1

    if not any([args.all, args.synthetic, args.historical, args.repo]):
        parser.print_help()
        return 0

    results = {}

    if args.all or args.synthetic:
        results["synthetic"] = SyntheticExampleGenerator().seed()

    if args.all or args.historical:
        results["historical"] = HistoricalIndexer().index_all()

    if args.repo:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        results["external_repo"] = ExternalRepoSeeder(
            source=args.repo,
            ref=args.ref,
            tags=tags,
            name=args.name,
        ).seed()

    print("种子注入结果:")
    for key, value in results.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
