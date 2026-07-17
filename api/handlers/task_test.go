package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/mingrun/resume-agent-system/api/models"
	"github.com/mingrun/resume-agent-system/api/service"
	"github.com/mingrun/resume-agent-system/api/store"
)

func setupTestServer(t *testing.T) (*gin.Engine, *store.RedisStore) {
	redisURL := os.Getenv("REDIS_URL")
	if redisURL == "" {
		redisURL = "redis://localhost:6379/0"
	}

	s, err := store.NewRedisStore(redisURL)
	if err != nil {
		t.Skipf("redis not available: %v", err)
	}

	gin.SetMode(gin.TestMode)
	r := gin.New()
	// 测试用简单中间件，模拟 auth 中间件注入 api_key
	r.Use(func(c *gin.Context) {
		key := c.GetHeader("X-API-Key")
		if key == "" {
			key = "test-key"
		}
		c.Set("api_key", key)
		c.Next()
	})
	svc := service.NewTaskService(s, nil)
	h := NewTaskHandler(svc)
	r.POST("/api/v1/tasks", h.CreateTask)
	r.GET("/api/v1/tasks/:id", h.GetTask)
	r.POST("/api/v1/tasks/:id/review", h.SubmitReview)
	r.POST("/api/v1/tasks/:id/cancel", h.CancelTask)
	r.GET("/api/v1/usage", h.GetUsage)

	return r, s
}

func TestCreateTask(t *testing.T) {
	r, _ := setupTestServer(t)

	reqBody, _ := json.Marshal(models.CreateTaskRequest{Requirements: "开发一个计算器"})
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/v1/tasks", bytes.NewBuffer(reqBody))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)

	if w.Code != http.StatusAccepted {
		t.Errorf("expected 202, got %d", w.Code)
	}

	var resp models.CreateTaskResponse
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp.TaskID == "" {
		t.Error("expected task_id")
	}
	if resp.Status != "pending" {
		t.Errorf("expected status pending, got %s", resp.Status)
	}
}

func TestGetTaskNotFound(t *testing.T) {
	r, _ := setupTestServer(t)

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/api/v1/tasks/not-exist", nil)
	r.ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestSubmitReview(t *testing.T) {
	r, _ := setupTestServer(t)

	reqBody, _ := json.Marshal(models.HumanReviewRequest{Feedback: "继续"})
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/v1/tasks/task-123/review", bytes.NewBuffer(reqBody))
	req.Header.Set("Content-Type", "application/json")
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestCancelTask(t *testing.T) {
	r, s := setupTestServer(t)
	ctx := context.Background()

	if err := s.CreateTask(ctx, "task-cancel-test", "测试取消", ""); err != nil {
		t.Fatalf("create task failed: %v", err)
	}

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("POST", "/api/v1/tasks/task-cancel-test/cancel", nil)
	r.ServeHTTP(w, req)

	if w.Code != http.StatusAccepted {
		t.Errorf("expected 202, got %d", w.Code)
	}

	data, err := s.GetTask(ctx, "task-cancel-test")
	if err != nil {
		t.Fatalf("get task failed: %v", err)
	}
	if data["status"] != "cancelling" {
		t.Errorf("expected status cancelling, got %s", data["status"])
	}
}

func TestGetUsage(t *testing.T) {
	r, s := setupTestServer(t)
	ctx := context.Background()

	if err := s.IncrementQuota(ctx, "key-usage", 150, 0.0123); err != nil {
		t.Fatalf("increment quota failed: %v", err)
	}

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/api/v1/usage", nil)
	req.Header.Set("X-API-Key", "key-usage")
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp["api_key"] != "key-usage" {
		t.Errorf("expected api_key key-usage, got %v", resp["api_key"])
	}
}

func TestGetTaskResultParsed(t *testing.T) {
	r, s := setupTestServer(t)

	// 直接写入一条带 JSON result 字符串的任务
	ctx := context.Background()
	if err := s.CreateTask(ctx, "task-result-test", "测试需求", ""); err != nil {
		t.Fatalf("create task failed: %v", err)
	}
	resultJSON := `{"overall_score":85,"passed":true}`
	if err := s.SetTaskResult(ctx, "task-result-test", resultJSON); err != nil {
		t.Fatalf("set result failed: %v", err)
	}

	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/api/v1/tasks/task-result-test", nil)
	r.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	result, ok := resp["result"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected result to be object, got %T", resp["result"])
	}
	if result["overall_score"] != float64(85) {
		t.Errorf("expected overall_score 85, got %v", result["overall_score"])
	}
	if result["passed"] != true {
		t.Errorf("expected passed true, got %v", result["passed"])
	}
}
