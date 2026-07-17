package config

import (
	"os"
	"strconv"
	"strings"
	"time"
)

// Config 保存 Gateway 配置
type Config struct {
	Port               string
	RedisURL           string
	APIKeys            []string
	RateLimitRPS       float64
	RateLimitBurst     int
	LogLevel           string
	EnableMetrics      bool
	CORSAllowedOrigins []string
	TLSCert            string
	TLSKey             string
	ShutdownTimeout    time.Duration
	QuotaMaxTokens     int64
	QuotaMaxCost       float64
}

// Load 从环境变量加载配置
func Load() *Config {
	return &Config{
		Port:               getEnv("PORT", "8080"),
		RedisURL:           getEnv("REDIS_URL", "redis://localhost:6379/0"),
		APIKeys:            splitKeys(getEnv("API_KEY", "")),
		RateLimitRPS:       parseFloat(getEnv("RATE_LIMIT_RPS", "10"), 10),
		RateLimitBurst:     parseInt(getEnv("RATE_LIMIT_BURST", "20"), 20),
		LogLevel:           getEnv("LOG_LEVEL", "info"),
		EnableMetrics:      parseBool(getEnv("ENABLE_METRICS", "true"), true),
		CORSAllowedOrigins: splitOrigins(getEnv("CORS_ALLOWED_ORIGINS", "*")),
		TLSCert:            getEnv("TLS_CERT", ""),
		TLSKey:             getEnv("TLS_KEY", ""),
		ShutdownTimeout:    parseDuration(getEnv("SHUTDOWN_TIMEOUT", "30s"), 30*time.Second),
		QuotaMaxTokens:     parseInt64(getEnv("QUOTA_MAX_TOKENS", "0"), 0),
		QuotaMaxCost:       parseFloat(getEnv("QUOTA_MAX_COST", "0"), 0),
	}
}

func getEnv(key, defaultValue string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return defaultValue
}

func splitKeys(v string) []string {
	if v == "" {
		return nil
	}
	parts := strings.Split(v, ",")
	var keys []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			keys = append(keys, p)
		}
	}
	return keys
}

func splitOrigins(v string) []string {
	if v == "" {
		return []string{"*"}
	}
	parts := strings.Split(v, ",")
	var origins []string
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			origins = append(origins, p)
		}
	}
	return origins
}

func parseInt(v string, defaultValue int) int {
	n, err := strconv.Atoi(v)
	if err != nil {
		return defaultValue
	}
	return n
}

func parseInt64(v string, defaultValue int64) int64 {
	n, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		return defaultValue
	}
	return n
}

func parseFloat(v string, defaultValue float64) float64 {
	n, err := strconv.ParseFloat(v, 64)
	if err != nil {
		return defaultValue
	}
	return n
}

func parseBool(v string, defaultValue bool) bool {
	switch strings.ToLower(v) {
	case "true", "1", "yes":
		return true
	case "false", "0", "no":
		return false
	default:
		return defaultValue
	}
}

func parseDuration(v string, defaultValue time.Duration) time.Duration {
	d, err := time.ParseDuration(v)
	if err != nil {
		return defaultValue
	}
	return d
}
