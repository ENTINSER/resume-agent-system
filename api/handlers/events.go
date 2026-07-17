package handlers

import (
	"encoding/json"
	"io"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/mingrun/resume-agent-system/api/store"
)

// EventHandler SSE 事件 handler
type EventHandler struct {
	store *store.RedisStore
}

// NewEventHandler 创建 EventHandler
func NewEventHandler(store *store.RedisStore) *EventHandler {
	return &EventHandler{store: store}
}

// TaskEvents SSE 任务事件流
func (h *EventHandler) TaskEvents(c *gin.Context) {
	id := c.Param("id")

	c.Stream(func(w io.Writer) bool {
		ctx := c.Request.Context()
		pubsub := h.store.SubscribeEvents(ctx)
		defer pubsub.Close()

		ch := pubsub.Channel()
		heartbeat := time.NewTicker(15 * time.Second)
		defer heartbeat.Stop()

		for {
			select {
			case <-ctx.Done():
				return false

			case <-heartbeat.C:
				c.SSEvent("heartbeat", map[string]string{"ts": time.Now().Format(time.RFC3339)})
				c.Writer.Flush()

			case msg, ok := <-ch:
				if !ok {
					return false
				}

				var event map[string]interface{}
				if err := json.Unmarshal([]byte(msg.Payload), &event); err != nil {
					continue
				}

				eventTaskID, _ := event["task_id"].(string)
				if eventTaskID != id {
					continue
				}

				c.SSEvent("message", event)
				c.Writer.Flush()

				if status, ok := event["status"].(string); ok && (status == "completed" || status == "failed") {
					return false
				}
			}
		}
	})
}
