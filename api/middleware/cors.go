package middleware

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/mingrun/resume-agent-system/api/config"
)

// CORS 可配置跨域中间件
func CORS(cfg *config.Config) gin.HandlerFunc {
	return func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")
		allowed := isAllowedOrigin(origin, cfg.CORSAllowedOrigins)

		if allowed {
			c.Writer.Header().Set("Access-Control-Allow-Origin", origin)
		} else if contains(cfg.CORSAllowedOrigins, "*") {
			c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		}

		c.Writer.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, X-Request-ID")
		c.Writer.Header().Set("Access-Control-Expose-Headers", "X-Request-ID")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	}
}

func isAllowedOrigin(origin string, allowed []string) bool {
	if origin == "" {
		return false
	}
	for _, o := range allowed {
		if o == origin || o == "*" {
			return true
		}
		if strings.HasSuffix(origin, o) && strings.HasPrefix(origin, ".") == false {
			// 支持子域名匹配，如 .example.com
			if strings.HasPrefix(o, ".") {
				domain := strings.TrimPrefix(o, ".")
				if strings.HasSuffix(origin, "."+domain) || origin == domain {
					return true
				}
			}
		}
	}
	return false
}

