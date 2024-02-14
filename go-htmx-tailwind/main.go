package main

import (
	"context"
	"fmt"
	"html/template"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"time"

	"github.com/ryanjdick/go-htmx-tailwind/app/handlers"
	"github.com/ryanjdick/go-htmx-tailwind/app/middleware"
)

type config struct {
	Host string
	Port string
}

func run(ctx context.Context, logger *slog.Logger, cfg config) error {
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	tmpl, err := template.ParseFiles("templates/hello.html")
	if err != nil {
		return err
	}

	http.Handle("GET /hello/{name}", middleware.WithLogging(logger, handlers.BuildGetHelloHandler(tmpl)))
	http.Handle("GET /time", middleware.WithLogging(logger, handlers.BuildGetTimeHandler(tmpl)))
	http.Handle("GET /", middleware.WithLogging(logger, http.FileServer(http.Dir("static"))))

	addr := ":8080"
	logger.InfoContext(ctx, fmt.Sprintf("Starting server at %v", addr))

	httpServer := &http.Server{
		Addr:        net.JoinHostPort(cfg.Host, cfg.Port),
		BaseContext: func(net.Listener) context.Context { return ctx },
	}

	go func() {
		logger.Info(fmt.Sprintf("listening on %s\n", httpServer.Addr))
		err := httpServer.ListenAndServe()
		if err != nil && err != http.ErrServerClosed {
			logger.Error(fmt.Sprintf("error listening and serving: %s\n", err))
		}
	}()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-ctx.Done()
		// Make a new context for the shutdown. We attempt to shutdown cleanly
		// by giving in-progress requests time to complete.
		shutdownCtx := context.Background()
		shutdownCtx, cancel := context.WithTimeout(shutdownCtx, 10*time.Second)
		defer cancel()
		if err := httpServer.Shutdown(shutdownCtx); err != nil {
			logger.Error(fmt.Sprintf("error shutting down http server: %s\n", err))
		}
	}()
	wg.Wait()

	return nil
}

func main() {
	ctx := context.Background()
	logger := slog.Default()
	cfg := config{
		Host: "localhost",
		Port: "8080",
	}
	if err := run(ctx, logger, cfg); err != nil {
		logger.Error(fmt.Sprintf("server error: %v", err))
		os.Exit(1)
	}
}
