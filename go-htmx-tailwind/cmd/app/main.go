package main

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path"
	"sync"
	"text/template"
	"time"

	"github.com/ryanjdick/go-htmx-tailwind/internal/handlers"
	"github.com/ryanjdick/go-htmx-tailwind/internal/middleware"
	"github.com/ryanjdick/go-htmx-tailwind/internal/utils"
)

type config struct {
	Host         string
	Port         string
	viteBuildDir string
}

func run(ctx context.Context, logger *slog.Logger, cfg config) error {
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	envConfig, err := utils.LoadEnv()
	if err != nil {
		return fmt.Errorf("failed to load env: %w", err)
	}

	// Prepare TemplateExecutor.
	var tmpl utils.TemplateExecutor
	templateFilenames := "templates/hello.html"
	if envConfig.Environment == utils.EEDevelopment {
		tmpl = utils.NewDevTemplate(templateFilenames)
	} else {
		tmpl, err = template.ParseFiles(templateFilenames)
		if err != nil {
			return fmt.Errorf("failed to parse template files: %w", err)
		}
	}

	viteManifest, err := utils.LoadViteManifestFromFile(path.Join(cfg.viteBuildDir, ".vite/manifest.json"))
	if err != nil {
		return fmt.Errorf("failed to load Vite manifest: %w", err)
	}

	http.Handle(
		"GET /hello/{name}",
		middleware.WithLogging(
			logger,
			handlers.BuildGetHelloHandler(tmpl, envConfig, "/"+viteManifest.MainJSFile.File, "/"+viteManifest.MainCSSFile.File),
		),
	)
	http.Handle(
		"GET /time",
		middleware.WithLogging(
			logger,
			handlers.BuildGetTimeHandler(tmpl),
		),
	)
	http.Handle(
		"GET /assets/",
		middleware.WithLogging(
			logger,
			http.StripPrefix("/assets/", http.FileServer(http.Dir(path.Join(cfg.viteBuildDir, "assets")))),
		),
	)
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

	logger.Info("shutting down...")

	return nil
}

func main() {
	ctx := context.Background()
	logger := slog.Default()
	cfg := config{
		Host:         "localhost",
		Port:         "8080",
		viteBuildDir: "frontend/dist",
	}
	if err := run(ctx, logger, cfg); err != nil {
		logger.Error(fmt.Sprintf("server error: %v", err))
		os.Exit(1)
	}
}
