# K8s 部署说明

```bash
# 1. 构建并推送镜像（请替换为你的镜像仓库）
docker build -t agent-platform-worker:latest -f Dockerfile.worker .
docker build -t agent-platform-api:latest ./api

# 2. 部署到集群
kubectl apply -f k8s/

# 3. 设置敏感配置
kubectl set env deployment/api KIMI_API_KEY=<key> API_KEY=<key> -n agent-platform
kubectl set env deployment/worker KIMI_API_KEY=<key> -n agent-platform

# 4. 执行数据库迁移
kubectl exec -it deployment/worker -n agent-platform -- alembic upgrade head
```
