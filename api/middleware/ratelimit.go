package middleware

import (
	"net/http"
	"sync"

	"github.com/gin-gonic/gin"
	"golang.org/x/time/rate"
	"github.com/mingrun/resume-agent-system/api/config"
	"github.com/mingrun/resume-agent-system/api/models"
)

// RateLimiter 令牌桶限流器
type RateLimiter struct {
	globalLimiter *rate.Limiter
	limiters      map[string]*rate.Limiter
	mu            sync.RWMutex
	r             rate.Limit
	b             int
}

// NewRateLimiter 创建限流器
func NewRateLimiter(cfg *config.Config) *RateLimiter {
	return &RateLimiter{
		globalLimiter: rate.NewLimiter(rate.Limit(cfg.RateLimitRPS), cfg.RateLimitBurst),
		limiters:      make(map[string]*rate.Limiter),
		r:             rate.Limit(cfg.RateLimitRPS),
		b:             cfg.RateLimitBurst,
	}
}

// Limit 全局限流中间件
func (rl *RateLimiter) Limit() gin.HandlerFunc {
	return func(c *gin.Context) {
		if !rl.globalLimiter.Allow() {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, models.ErrorResponse{
				Error:     "rate limit exceeded",
				Code:      "rate_limited",
				RequestID: c.GetString("request_id"),
			})
			return
		}
		c.Next()
	}
}

// LimitByKey 按 Key 限流中间件
func (rl *RateLimiter) LimitByKey() gin.HandlerFunc {
	return func(c *gin.Context) {
		key := c.GetHeader("X-API-Key")
		if key == "" {
			key = c.ClientIP()
		}

		limiter := rl.getLimiter(key)
		if !limiter.Allow() {
			c.AbortWithStatusJSON(http.StatusTooManyRequests, models.ErrorResponse{
				Error:     "rate limit exceeded for key",
				Code:      "rate_limited",
				RequestID: c.GetString("request_id"),
			})
			return
		}
		c.Next()
	}
}

func (rl *RateLimiter) getLimiter(key string) *rate.Limiter {
	rl.mu.RLock()
	limiter, ok := rl.limiters[key]
	rl.mu.RUnlock()

	if ok {
		return limiter
	}

	rl.mu.Lock()
	defer rl.mu.Unlock()

	limiter, ok = rl.limiters[key]
	if ok {
		return limiter
	}

	limiter = rate.NewLimiter(rl.r, rl.b)
	rl.limiters[key] = limiter
	return limiter
}
