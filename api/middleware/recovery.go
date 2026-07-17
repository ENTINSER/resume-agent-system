package middleware

import (
	"fmt"
	"log/slog"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/mingrun/resume-agent-system/api/models"
)

// Recovery panic 恢复中间件
func Recovery(logger *slog.Logger) gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, err interface{}) {
		logger.Error("panic recovered",
			slog.String("request_id", c.GetString("request_id")),
			slog.String("path", c.Request.URL.Path),
			slog.String("error", fmt.Sprintf("%v", err)),
		)
		c.AbortWithStatusJSON(http.StatusInternalServerError, models.ErrorResponse{
			Error:     "internal server error",
			Code:      "internal_error",
			RequestID: c.GetString("request_id"),
		})
	})
}
