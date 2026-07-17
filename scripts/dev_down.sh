#!/usr/bin/env bash
# 停止本地开发依赖容器
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

echo "停止开发依赖容器..."
docker compose -f docker-compose.dev.yml down

echo "开发环境已停止"
