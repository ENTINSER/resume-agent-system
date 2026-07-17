package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/mingrun/resume-agent-system/api/models"
	"github.com/mingrun/resume-agent-system/api/service"
)

// TaskHandler 任务 handler
type TaskHandler struct {
	service *service.TaskService
}

// NewTaskHandler 创建 TaskHandler
func NewTaskHandler(service *service.TaskService) *TaskHandler {
	return &TaskHandler{service: service}
}

// CreateTask 创建任务
func (h *TaskHandler) CreateTask(c *gin.Context) {
	var req models.CreateTaskRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, "invalid request", "bad_request")
		return
	}

	apiKey, _ := c.Get("api_key")
	keyStr, _ := apiKey.(string)

	if err := h.service.CheckQuota(c.Request.Context(), keyStr); err != nil {
		respondError(c, http.StatusTooManyRequests, err.Error(), "quota_exceeded")
		return
	}

	resp, err := h.service.CreateTask(c.Request.Context(), req, keyStr)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err.Error(), "internal_error")
		return
	}

	c.JSON(http.StatusAccepted, resp)
}

// GetTask 查询任务
func (h *TaskHandler) GetTask(c *gin.Context) {
	id := c.Param("id")
	data, err := h.service.GetTask(c.Request.Context(), id)
	if err != nil {
		respondError(c, http.StatusNotFound, "task not found", "not_found")
		return
	}

	// 把 string map 转为 interface{} map，以便把 result JSON 字符串反序列化为对象
	resp := make(map[string]interface{}, len(data))
	for k, v := range data {
		resp[k] = v
	}

	if resultStr, ok := data["result"]; ok && resultStr != "" {
		var resultObj interface{}
		if err := json.Unmarshal([]byte(resultStr), &resultObj); err == nil {
			resp["result"] = resultObj
		}
		// 反序列化失败时保留原字符串，由前端自行处理
	}

	c.JSON(http.StatusOK, resp)
}

// ListTasks 列出任务
func (h *TaskHandler) ListTasks(c *gin.Context) {
	tasks, err := h.service.ListTasks(c.Request.Context())
	if err != nil {
		respondError(c, http.StatusInternalServerError, err.Error(), "internal_error")
		return
	}

	c.JSON(http.StatusOK, gin.H{"tasks": tasks, "total": len(tasks)})
}

// SubmitReview 提交人类反馈
func (h *TaskHandler) SubmitReview(c *gin.Context) {
	id := c.Param("id")
	var req models.HumanReviewRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		respondError(c, http.StatusBadRequest, "invalid request", "bad_request")
		return
	}

	if err := h.service.SubmitReview(c.Request.Context(), id, req); err != nil {
		respondError(c, http.StatusInternalServerError, err.Error(), "internal_error")
		return
	}

	c.JSON(http.StatusOK, gin.H{"task_id": id, "feedback": req.Feedback})
}

// CancelTask 取消任务
func (h *TaskHandler) CancelTask(c *gin.Context) {
	id := c.Param("id")

	if err := h.service.CancelTask(c.Request.Context(), id); err != nil {
		respondError(c, http.StatusInternalServerError, err.Error(), "internal_error")
		return
	}

	c.JSON(http.StatusAccepted, gin.H{"task_id": id, "status": "cancelling"})
}

// GetUsage 查询当前 API Key 的累计用量
func (h *TaskHandler) GetUsage(c *gin.Context) {
	apiKey, _ := c.Get("api_key")
	keyStr, _ := apiKey.(string)
	if keyStr == "" {
		respondError(c, http.StatusUnauthorized, "missing api key", "unauthorized")
		return
	}

	usage, err := h.service.GetQuota(c.Request.Context(), keyStr)
	if err != nil {
		respondError(c, http.StatusInternalServerError, err.Error(), "internal_error")
		return
	}

	if usage == nil {
		usage = map[string]string{}
	}
	c.JSON(http.StatusOK, gin.H{
		"api_key":      keyStr,
		"total_tokens": usage["total_tokens"],
		"total_cost":   usage["total_cost"],
		"task_count":   usage["task_count"],
	})
}

func respondError(c *gin.Context, status int, message, code string) {
	c.AbortWithStatusJSON(status, models.ErrorResponse{
		Error:     message,
		Code:      code,
		RequestID: c.GetString("request_id"),
	})
}
