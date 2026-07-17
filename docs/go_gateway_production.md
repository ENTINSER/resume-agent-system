# Go API Gateway 生产化

## 1. 架构

```
                    ┌─────────────────────┐
                    │   Load Balancer     │
                    └──────────┬──────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Gateway 1  │     │  Gateway 2  │     │  Gateway N  │
    │  (stateless)│     │  (stateless)│     │  (stateless)│
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               ▼
                         ┌─────────────┐
                         │    Redis    │
                         │   + Celery  │
                         └─────────────┘
```

Gateway 为无状态设计，所有状态存储在 Redis，可水平扩展。

## 2. 目录结构

```
api/
├── main.go              # 启动入口、优雅关闭
├── config/
│   └── config.go        # 配置加载
├── middleware/
│   ├── auth.go          # API Key 认证
│   ├── ratelimit.go     # 令牌桶限流
│   ├── logger.go        # 结构化日志 + request_id
│   ├── metrics.go       # Prometheus 指标
│   ├── recovery.go      # Panic 恢复
│   └── cors.go          # 跨域配置
├── handlers/
│   ├── task.go          # 任务接口
│   ├── events.go        # SSE 事件流
│   └── health.go        # 健康检查
├── service/
│   └── task_service.go  # 业务逻辑
├── store/
│   └── redis.go         # Redis 封装
└── static/              # 前端资源
```

## 3. 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PORT` | 监听端口 | `8080` |
| `REDIS_URL` | Redis 地址 | `redis://localhost:6379/0` |
| `API_KEY` | API Key（逗号分隔多个） | 空（不认证） |
| `RATE_LIMIT_RPS` | 令牌桶每秒填充速率 | `10` |
| `RATE_LIMIT_BURST` | 令牌桶容量 | `20` |
| `LOG_LEVEL` | 日志级别 | `info` |
| `ENABLE_METRICS` | 启用 Prometheus 指标 | `true` |
| `CORS_ALLOWED_ORIGINS` | 允许跨域来源 | `*` |
| `TLS_CERT` | TLS 证书路径 | 空 |
| `TLS_KEY` | TLS 私钥路径 | 空 |
| `SHUTDOWN_TIMEOUT` | 优雅关闭超时 | `30s` |

## 4. 接口列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/tasks` | 创建任务 |
| GET | `/api/v1/tasks` | 任务列表 |
| GET | `/api/v1/tasks/:id` | 查询任务 |
| GET | `/api/v1/tasks/:id/events` | SSE 事件流 |
| POST | `/api/v1/tasks/:id/review` | 提交人类反馈 |
| GET | `/healthz` | 综合健康 |
| GET | `/livez` | 存活探针 |
| GET | `/readyz` | 就绪探针 |
| GET | `/metrics` | Prometheus 指标 |

## 5. 认证

支持两种方式：

```bash
# Authorization Bearer
curl -H "Authorization: Bearer <API_KEY>" http://localhost:8080/api/v1/tasks

# X-API-Key
curl -H "X-API-Key: <API_KEY>" http://localhost:8080/api/v1/tasks
```

## 6. 限流

- 全局限流：`RATE_LIMIT_RPS` / `RATE_LIMIT_BURST`
- 按 Key 限流：基于 `X-API-Key` 或客户端 IP
- 超过限制返回 429 Too Many Requests

## 7. 可观测性

### 日志

每条请求输出 JSON 日志：

```json
{
  "time": "2026-06-30T15:30:00Z",
  "level": "INFO",
  "msg": "http_request",
  "request_id": "uuid",
  "method": "POST",
  "path": "/api/v1/tasks",
  "status": 202,
  "duration": "1.234ms",
  "client_ip": "127.0.0.1"
}
```

### Prometheus 指标

- `http_requests_total{method, path, status}`
- `http_request_duration_seconds{method, path}`

## 8. K8s 部署示例

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agent-gateway
  template:
    metadata:
      labels:
        app: agent-gateway
    spec:
      containers:
      - name: gateway
        image: agent-gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_URL
          value: "redis://redis:6379/0"
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: agent-api-key
              key: key
        - name: RATE_LIMIT_RPS
          value: "100"
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /livez
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 15
---
apiVersion: v1
kind: Service
metadata:
  name: agent-gateway
spec:
  selector:
    app: agent-gateway
  ports:
  - port: 80
    targetPort: 8080
```

## 9. TLS

```bash
# 生成自签名证书（测试用）
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# 启动
TLS_CERT=cert.pem TLS_KEY=key.pem ./agent-api
```

## 10. 本地开发

```bash
cd api
GOPROXY=https://goproxy.cn,direct go build -o agent-api .
./agent-api
```

## 11. 测试

```bash
cd api
GOPROXY=https://goproxy.cn,direct go test ./...
```
