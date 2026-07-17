package handlers

import (
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/mingrun/resume-agent-system/api/store"
)

func setupHealthServer(t *testing.T) (*gin.Engine, *store.RedisStore) {
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
	h := NewHealthHandler(s)
	r.GET("/livez", h.Live)
	r.GET("/readyz", h.Ready)
	r.GET("/healthz", h.Health)

	return r, s
}

func TestLive(t *testing.T) {
	r, _ := setupHealthServer(t)
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/livez", nil)
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestReady(t *testing.T) {
	r, _ := setupHealthServer(t)
	w := httptest.NewRecorder()
	req, _ := http.NewRequest("GET", "/readyz", nil)
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}
