package middleware

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/mingrun/resume-agent-system/api/config"
	"github.com/mingrun/resume-agent-system/api/models"
)

// APIKeyAuth API Key 认证中间件
func APIKeyAuth(cfg *config.Config) gin.HandlerFunc {
	return func(c *gin.Context) {
		if len(cfg.APIKeys) == 0 {
			c.Next()
			return
		}

		key := extractAPIKey(c)
		if key == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, models.ErrorResponse{
				Error:     "missing api key",
				Code:      "unauthorized",
				RequestID: c.GetString("request_id"),
			})
			return
		}

		if !contains(cfg.APIKeys, key) {
			c.AbortWithStatusJSON(http.StatusUnauthorized, models.ErrorResponse{
				Error:     "invalid api key",
				Code:      "unauthorized",
				RequestID: c.GetString("request_id"),
			})
			return
		}

		c.Set("api_key", key)
		c.Next()
	}
}

func extractAPIKey(c *gin.Context) string {
	auth := c.GetHeader("Authorization")
	if auth != "" {
		parts := strings.SplitN(auth, " ", 2)
		if len(parts) == 2 && strings.EqualFold(parts[0], "Bearer") {
			return parts[1]
		}
	}

	if key := c.GetHeader("X-API-Key"); key != "" {
		return key
	}

	return c.Query("api_key")
}

func contains(arr []string, target string) bool {
	for _, v := range arr {
		if v == target {
			return true
		}
	}
	return false
}
