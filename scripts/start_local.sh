#!/bin/bash
# 本地开发环境启动脚本

set -e

cd "$(dirname "$0")/.."

# 1. 启动 Redis（如果未运行）
if ! redis-cli ping > /dev/null 2>&1; then
    echo "启动 Redis..."
    redis-server --daemonize yes
else
    echo "Redis 已在运行"
fi

# 2. 启动 Qdrant（如果未运行）
if ! curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "尝试启动 Docker Qdrant..."
    if docker run -d --name qdrant-local -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant:latest > /dev/null 2>&1; then
        echo "Qdrant Docker 已启动: http://localhost:6333"
    else
        echo "Docker Qdrant 启动失败，回退到本地文件模式: data/qdrant_storage"
        export QDRANT_URL="data/qdrant_storage"
    fi
else
    echo "Qdrant 已在运行"
fi

# 3. 初始化数据库
source venv/bin/activate
python -c "from src.db import init_db; init_db()"

# 4. 启动 Celery Worker（后台）
echo "启动 Celery Worker..."
# macOS 上 prefork + PyTorch/Metal 在 fork 后加载会 SIGABRT，使用 solo 模式
if [[ "$OSTYPE" == "darwin"* ]]; then
    nohup venv/bin/celery -A src.tasks worker -l info -P solo > logs/celery.log 2>&1 &
else
    nohup venv/bin/celery -A src.tasks worker -l info -c 2 > logs/celery.log 2>&1 &
fi
echo $! > logs/celery.pid

# 5. 启动 Go API（后台）
echo "启动 Go API..."
cd api
nohup ./agent-api > ../logs/api.log 2>&1 &
echo $! > ../logs/api.pid
cd ..

QDRANT_ECHO=${QDRANT_URL:-http://localhost:6333}
echo "本地服务已启动："
echo "  Redis:    redis://localhost:6379"
echo "  Qdrant:   $QDRANT_ECHO"
echo "  API:      http://localhost:8080"
echo "  UI:       http://localhost:8080/"
echo ""
echo "查看日志： tail -f logs/celery.log logs/api.log"
