#!/bin/bash
# 停止本地开发环境

cd "$(dirname "$0")/.."

if [ -f logs/celery.pid ]; then
    echo "停止 Celery Worker..."
    kill $(cat logs/celery.pid) 2>/dev/null || true
    rm logs/celery.pid
fi

if [ -f logs/api.pid ]; then
    echo "停止 Go API..."
    kill $(cat logs/api.pid) 2>/dev/null || true
    rm logs/api.pid
fi

# 停止 Qdrant Docker 容器
if docker ps -q -f name=qdrant-local > /dev/null 2>&1; then
    echo "停止 Qdrant..."
    docker stop qdrant-local > /dev/null 2>&1 || true
fi

echo "本地服务已停止"
