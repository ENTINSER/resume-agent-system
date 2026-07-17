package models

// CreateTaskRequest 创建任务请求
type CreateTaskRequest struct {
	Requirements string `json:"requirements" binding:"required,max=5000"`
}

// CreateTaskResponse 创建任务响应
type CreateTaskResponse struct {
	TaskID string `json:"task_id"`
	Status string `json:"status"`
}

// HumanReviewRequest 人类反馈请求
type HumanReviewRequest struct {
	Feedback string `json:"feedback" binding:"required,max=5000"`
}

// ErrorResponse 统一错误响应
type ErrorResponse struct {
	Error     string `json:"error"`
	Code      string `json:"code,omitempty"`
	RequestID string `json:"request_id,omitempty"`
}

// TaskEvent 任务事件
type TaskEvent struct {
	TaskID    string                 `json:"task_id"`
	Status    string                 `json:"status,omitempty"`
	Phase     string                 `json:"phase,omitempty"`
	Payload   map[string]interface{} `json:"payload,omitempty"`
	Timestamp int64                  `json:"timestamp"`
}
