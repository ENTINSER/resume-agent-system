#!/usr/bin/env bash
# 启动本地开发依赖（Redis / Postgres / Qdrant）
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "启动开发依赖容器..."
docker compose -f docker-compose.dev.yml up -d

echo "等待服务健康检查通过..."
docker compose -f docker-compose.dev.yml ps

echo ""
echo "开发环境已启动:"
echo "  Redis:    redis://localhost:6379/0"
echo "  Postgres: postgresql://agent:agent123@localhost:5432/agent_platform"
echo "  Qdrant:   http://localhost:6333"
echo ""
echo "停止请运行: ./scripts/dev_down.sh"
