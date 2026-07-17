package handlers

import (
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/mingrun/resume-agent-system/api/store"
)

// HealthHandler 健康检查 handler
type HealthHandler struct {
	store *store.RedisStore
}

// NewHealthHandler 创建 HealthHandler
func NewHealthHandler(store *store.RedisStore) *HealthHandler {
	return &HealthHandler{store: store}
}

// Live 存活检查
func (h *HealthHandler) Live(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status": "ok",
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

// Ready 就绪检查
func (h *HealthHandler) Ready(c *gin.Context) {
	if err := h.store.Ping(c.Request.Context()); err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"status": "not_ready",
			"redis":  false,
			"error":  err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"status": "ready",
		"redis":  true,
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}

// Health 综合健康检查
func (h *HealthHandler) Health(c *gin.Context) {
	redisOK := h.store.Ping(c.Request.Context()) == nil
	status := http.StatusOK
	if !redisOK {
		status = http.StatusServiceUnavailable
	}

	c.JSON(status, gin.H{
		"status": "ok",
		"redis":  redisOK,
		"time":   time.Now().UTC().Format(time.RFC3339),
	})
}
