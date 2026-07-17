package service

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/google/uuid"
	"github.com/mingrun/resume-agent-system/api/config"
	"github.com/mingrun/resume-agent-system/api/models"
	"github.com/mingrun/resume-agent-system/api/store"
)

// TaskService 任务业务逻辑
type TaskService struct {
	store *store.RedisStore
	cfg   *config.Config
}

// NewTaskService 创建 TaskService
func NewTaskService(store *store.RedisStore, cfg *config.Config) *TaskService {
	if cfg == nil {
		cfg = config.Load()
	}
	return &TaskService{store: store, cfg: cfg}
}

// CreateTask 创建任务
func (s *TaskService) CreateTask(ctx context.Context, req models.CreateTaskRequest, apiKey string) (*models.CreateTaskResponse, error) {
	taskID := uuid.New().String()

	if err := s.store.CreateTask(ctx, taskID, req.Requirements, apiKey); err != nil {
		return nil, fmt.Errorf("create task: %w", err)
	}

	// Celery 兼容的消息信封（kombu Redis transport 要求）
	messageID := uuid.New().String()

	// body: [[args...], {kwargs...}, {"callbacks":null,"errbacks":null,"chain":null,"chord":null}]
	celeryBody := []interface{}{
		[]interface{}{taskID, req.Requirements},
		map[string]interface{}{"api_key": apiKey},
		map[string]interface{}{
			"callbacks": nil,
			"errbacks":  nil,
			"chain":     nil,
			"chord":     nil,
		},
	}
	bodyBytes, err := json.Marshal(celeryBody)
	if err != nil {
		return nil, fmt.Errorf("marshal celery body: %w", err)
	}
	bodyB64 := base64.StdEncoding.EncodeToString(bodyBytes)

	headers := map[string]interface{}{
		"lang":                  "py",
		"task":                  "src.tasks.agent_task.run_agent_task",
		"id":                    messageID,
		"shadow":                nil,
		"eta":                   nil,
		"expires":               nil,
		"group":                 nil,
		"group_index":           nil,
		"retries":               0,
		"timelimit":             []interface{}{nil, nil},
		"root_id":               messageID,
		"parent_id":             nil,
		"argsrepr":              fmt.Sprintf("[%s, %s]", strconv.Quote(taskID), strconv.Quote(req.Requirements)),
		"kwargsrepr":            fmt.Sprintf("{%s: %s}", strconv.Quote("api_key"), strconv.Quote(apiKey)),
		"origin":                fmt.Sprintf("gen@%s", uuid.New().String()),
		"ignore_result":         true,
		"replaced_task_nesting": 0,
		"stamped_headers":       nil,
		"stamps":                map[string]interface{}{},
	}

	properties := map[string]interface{}{
		"correlation_id": messageID,
		"reply_to":       uuid.New().String(),
		"delivery_mode":  2,
		"delivery_info": map[string]interface{}{
			"exchange":    "",
			"routing_key": "celery",
		},
		"priority":      0,
		"body_encoding": "base64",
		"delivery_tag":  uuid.New().String(),
	}

	envelope := map[string]interface{}{
		"body":             bodyB64,
		"content-encoding": "utf-8",
		"content-type":     "application/json",
		"headers":          headers,
		"properties":       properties,
	}

	envelopeBytes, err := json.Marshal(envelope)
	if err != nil {
		return nil, fmt.Errorf("marshal celery envelope: %w", err)
	}

	if err := s.store.PublishCeleryTask(ctx, envelopeBytes); err != nil {
		return nil, fmt.Errorf("publish celery task: %w", err)
	}

	return &models.CreateTaskResponse{TaskID: taskID, Status: "pending"}, nil
}

// GetTask 查询任务
func (s *TaskService) GetTask(ctx context.Context, taskID string) (map[string]string, error) {
	return s.store.GetTask(ctx, taskID)
}

// ListTasks 列出任务
func (s *TaskService) ListTasks(ctx context.Context) ([]map[string]string, error) {
	return s.store.ListTasks(ctx)
}

// SubmitReview 提交人类反馈
func (s *TaskService) SubmitReview(ctx context.Context, taskID string, req models.HumanReviewRequest) error {
	return s.store.SubmitReview(ctx, taskID, req.Feedback)
}

// CancelTask 取消任务
func (s *TaskService) CancelTask(ctx context.Context, taskID string) error {
	return s.store.SetCancelFlag(ctx, taskID)
}

// GetQuota 查询 API Key 用量
func (s *TaskService) GetQuota(ctx context.Context, apiKey string) (map[string]string, error) {
	return s.store.GetQuota(ctx, apiKey)
}

// CheckQuota 检查 API Key 是否超出配额
func (s *TaskService) CheckQuota(ctx context.Context, apiKey string) error {
	if apiKey == "" {
		return nil
	}
	if s.cfg.QuotaMaxTokens <= 0 && s.cfg.QuotaMaxCost <= 0 {
		return nil
	}

	usage, err := s.store.GetQuota(ctx, apiKey)
	if err != nil {
		return err
	}

	if s.cfg.QuotaMaxTokens > 0 {
		tokens, _ := strconv.ParseInt(usage["total_tokens"], 10, 64)
		if tokens >= s.cfg.QuotaMaxTokens {
			return fmt.Errorf("quota exceeded: tokens")
		}
	}
	if s.cfg.QuotaMaxCost > 0 {
		cost, _ := strconv.ParseFloat(usage["total_cost"], 64)
		if cost >= s.cfg.QuotaMaxCost {
			return fmt.Errorf("quota exceeded: cost")
		}
	}
	return nil
}
