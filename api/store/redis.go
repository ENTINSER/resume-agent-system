package store

import (
	"context"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	redisKeyPrefix  = "agent:task:"
	redisEventsChan = "agent:task:events"
	celeryQueueName = "celery"
)

// RedisStore 封装 Redis 操作
type RedisStore struct {
	client *redis.Client
	ctx    context.Context
}

// NewRedisStore 创建 RedisStore
func NewRedisStore(redisURL string) (*RedisStore, error) {
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("parse redis url: %w", err)
	}

	client := redis.NewClient(opt)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis ping: %w", err)
	}

	return &RedisStore{
		client: client,
		ctx:    context.Background(),
	}, nil
}

// Close 关闭连接
func (s *RedisStore) Close() error {
	return s.client.Close()
}

// Ping 检查连接
func (s *RedisStore) Ping(ctx context.Context) error {
	return s.client.Ping(ctx).Err()
}

// CreateTask 创建任务记录（含创建者 API Key）
func (s *RedisStore) CreateTask(ctx context.Context, taskID, requirements, apiKey string) error {
	now := fmt.Sprintf("%d", time.Now().Unix())
	key := redisKeyPrefix + taskID
	return s.client.HSet(ctx, key, map[string]interface{}{
		"task_id":       taskID,
		"requirements":  requirements,
		"api_key":       apiKey,
		"status":        "pending",
		"current_phase": "init",
		"created_at":    now,
		"updated_at":    now,
		"result":        "",
		"error":         "",
	}).Err()
}

// GetTask 查询任务
func (s *RedisStore) GetTask(ctx context.Context, taskID string) (map[string]string, error) {
	key := redisKeyPrefix + taskID
	data, err := s.client.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("task not found")
	}
	return data, nil
}

// ListTasks 列出所有任务
func (s *RedisStore) ListTasks(ctx context.Context) ([]map[string]string, error) {
	iter := s.client.Scan(ctx, 0, redisKeyPrefix+"*", 100).Iterator()
	var keys []string
	for iter.Next(ctx) {
		keys = append(keys, iter.Val())
	}
	if err := iter.Err(); err != nil {
		return nil, err
	}

	var tasks []map[string]string
	for _, key := range keys {
		data, err := s.client.HGetAll(ctx, key).Result()
		if err != nil {
			continue
		}
		tasks = append(tasks, data)
	}
	return tasks, nil
}

// PublishCeleryTask 投递 Celery 任务
func (s *RedisStore) PublishCeleryTask(ctx context.Context, body []byte) error {
	return s.client.LPush(ctx, celeryQueueName, body).Err()
}

// SubmitReview 提交人类反馈
func (s *RedisStore) SubmitReview(ctx context.Context, taskID, feedback string) error {
	reviewKey := fmt.Sprintf("agent:human_review:%s", taskID)
	return s.client.Set(ctx, reviewKey, feedback, 24*time.Hour).Err()
}

// SetCancelFlag 设置任务取消标志
func (s *RedisStore) SetCancelFlag(ctx context.Context, taskID string) error {
	cancelKey := fmt.Sprintf("agent:cancel:%s", taskID)
	if err := s.client.Set(ctx, cancelKey, "1", 24*time.Hour).Err(); err != nil {
		return err
	}
	// 同步更新任务状态为 cancelling，便于前端即时感知
	key := redisKeyPrefix + taskID
	return s.client.HSet(ctx, key, map[string]interface{}{
		"status":        "cancelling",
		"current_phase": "cancelling",
		"updated_at":    fmt.Sprintf("%d", time.Now().Unix()),
	}).Err()
}

// SetTaskResult 设置任务结果（用于测试和后续 Go 端更新）
func (s *RedisStore) SetTaskResult(ctx context.Context, taskID, result string) error {
	key := redisKeyPrefix + taskID
	return s.client.HSet(ctx, key, map[string]interface{}{
		"result":     result,
		"updated_at": fmt.Sprintf("%d", time.Now().Unix()),
	}).Err()
}

// SubscribeEvents 订阅任务事件
func (s *RedisStore) SubscribeEvents(ctx context.Context) *redis.PubSub {
	return s.client.Subscribe(ctx, redisEventsChan)
}

// GetQuota 查询 API Key 的累计用量
func (s *RedisStore) GetQuota(ctx context.Context, apiKey string) (map[string]string, error) {
	key := fmt.Sprintf("agent:quota:%s", apiKey)
	return s.client.HGetAll(ctx, key).Result()
}

// IncrementQuota 增加 API Key 的累计用量
func (s *RedisStore) IncrementQuota(ctx context.Context, apiKey string, tokens int64, cost float64) error {
	key := fmt.Sprintf("agent:quota:%s", apiKey)
	pipe := s.client.Pipeline()
	pipe.HIncrBy(ctx, key, "total_tokens", tokens)
	pipe.HIncrBy(ctx, key, "task_count", 1)
	pipe.HSet(ctx, key, "total_cost", fmt.Sprintf("%.6f", cost))
	_, err := pipe.Exec(ctx)
	return err
}
