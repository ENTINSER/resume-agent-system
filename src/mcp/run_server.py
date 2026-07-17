"""MCP Server 独立进程入口

用法：
    python -m src.mcp.run_server --workspace projects/<session-id>
"""

import argparse
import asyncio

from src.mcp.server import create_mcp_server


def main():
    parser = argparse.ArgumentParser(description="Resume Agent MCP Server")
    parser.add_argument("--workspace", required=True, help="工作空间目录")
    parser.add_argument("--name", default="resume-agent-mcp", help="Server 名称")
    args = parser.parse_args()

    mcp = create_mcp_server(workspace=args.workspace, name=args.name)
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
