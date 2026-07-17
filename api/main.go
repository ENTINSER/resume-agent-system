package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/mingrun/resume-agent-system/api/config"
	"github.com/mingrun/resume-agent-system/api/handlers"
	"github.com/mingrun/resume-agent-system/api/middleware"
	"github.com/mingrun/resume-agent-system/api/service"
	"github.com/mingrun/resume-agent-system/api/store"
)

func main() {
	cfg := config.Load()
	logger := middleware.InitLogger(cfg.LogLevel)

	redisStore, err := store.NewRedisStore(cfg.RedisURL)
	if err != nil {
		logger.Error("failed to connect redis", slog.String("error", err.Error()))
		os.Exit(1)
	}
	defer redisStore.Close()

	taskService := service.NewTaskService(redisStore, cfg)
	taskHandler := handlers.NewTaskHandler(taskService)
	eventHandler := handlers.NewEventHandler(redisStore)
	healthHandler := handlers.NewHealthHandler(redisStore)

	rateLimiter := middleware.NewRateLimiter(cfg)

	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(middleware.Recovery(logger))
	r.Use(middleware.RequestLogger(logger))
	r.Use(middleware.MetricsMiddleware())
	r.Use(middleware.CORS(cfg))
	r.Use(middleware.APIKeyAuth(cfg))
	r.Use(rateLimiter.Limit())
	r.Use(rateLimiter.LimitByKey())

	// API routes
	r.POST("/api/v1/tasks", taskHandler.CreateTask)
	r.GET("/api/v1/tasks", taskHandler.ListTasks)
	r.GET("/api/v1/tasks/:id", taskHandler.GetTask)
	r.GET("/api/v1/tasks/:id/events", eventHandler.TaskEvents)
	r.POST("/api/v1/tasks/:id/review", taskHandler.SubmitReview)
	r.POST("/api/v1/tasks/:id/cancel", taskHandler.CancelTask)
	r.GET("/api/v1/usage", taskHandler.GetUsage)

	// Health & metrics
	r.GET("/healthz", healthHandler.Health)
	r.GET("/livez", healthHandler.Live)
	r.GET("/readyz", healthHandler.Ready)
	if cfg.EnableMetrics {
		r.GET("/metrics", middleware.PrometheusHandler())
	}

	// Static UI
	r.Static("/static", "./static")
	r.StaticFile("/", "./static/index.html")

	srv := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      r,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	go func() {
		logger.Info("api gateway listening", slog.String("addr", srv.Addr))
		var err error
		if cfg.TLSCert != "" && cfg.TLSKey != "" {
			err = srv.ListenAndServeTLS(cfg.TLSCert, cfg.TLSKey)
		} else {
			err = srv.ListenAndServe()
		}
		if err != nil && err != http.ErrServerClosed {
			logger.Error("server failed", slog.String("error", err.Error()))
			os.Exit(1)
		}
	}()

	// Graceful shutdown
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logger.Info("shutting down api gateway")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()

	if err := srv.Shutdown(shutdownCtx); err != nil {
		logger.Error("server shutdown error", slog.String("error", err.Error()))
	}
	redisStore.Close()
	logger.Info("api gateway stopped")
}
