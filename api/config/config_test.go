package config

import (
	"os"
	"testing"
	"time"
)

func TestLoadDefaults(t *testing.T) {
	cfg := Load()
	if cfg.Port != "8080" {
		t.Errorf("expected port 8080, got %s", cfg.Port)
	}
	if cfg.RateLimitRPS != 10 {
		t.Errorf("expected rps 10, got %f", cfg.RateLimitRPS)
	}
	if cfg.ShutdownTimeout != 30*time.Second {
		t.Errorf("expected shutdown timeout 30s, got %v", cfg.ShutdownTimeout)
	}
}

func TestLoadFromEnv(t *testing.T) {
	os.Setenv("PORT", "9090")
	os.Setenv("API_KEY", "key1,key2")
	os.Setenv("RATE_LIMIT_RPS", "100")
	defer func() {
		os.Unsetenv("PORT")
		os.Unsetenv("API_KEY")
		os.Unsetenv("RATE_LIMIT_RPS")
	}()

	cfg := Load()
	if cfg.Port != "9090" {
		t.Errorf("expected port 9090, got %s", cfg.Port)
	}
	if len(cfg.APIKeys) != 2 {
		t.Errorf("expected 2 api keys, got %d", len(cfg.APIKeys))
	}
	if cfg.RateLimitRPS != 100 {
		t.Errorf("expected rps 100, got %f", cfg.RateLimitRPS)
	}
}
