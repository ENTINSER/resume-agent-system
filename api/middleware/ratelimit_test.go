package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/mingrun/resume-agent-system/api/config"
)

func TestRateLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cfg := &config.Config{RateLimitRPS: 1, RateLimitBurst: 1}
	rl := NewRateLimiter(cfg)

	r := gin.New()
	r.Use(rl.Limit())
	r.GET("/test", func(c *gin.Context) {
		c.Status(http.StatusOK)
	})

	w1 := httptest.NewRecorder()
	req1, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w1, req1)
	if w1.Code != http.StatusOK {
		t.Errorf("first request expected 200, got %d", w1.Code)
	}

	w2 := httptest.NewRecorder()
	req2, _ := http.NewRequest("GET", "/test", nil)
	r.ServeHTTP(w2, req2)
	if w2.Code != http.StatusTooManyRequests {
		t.Errorf("second request expected 429, got %d", w2.Code)
	}
}
