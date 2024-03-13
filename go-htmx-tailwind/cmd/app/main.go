package main

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path"
	"path/filepath"
	"sync"
	"text/template"
	"time"

	_ "github.com/mattn/go-sqlite3"
	"github.com/ryanjdick/go-htmx-tailwind/internal/db_utils"
	"github.com/ryanjdick/go-htmx-tailwind/internal/handlers"
	"github.com/ryanjdick/go-htmx-tailwind/internal/middleware"
	"github.com/ryanjdick/go-htmx-tailwind/internal/utils"
)

func run(ctx context.Context, logger *slog.Logger, cfg *utils.AppConfig) error {
	ctx, cancel := signal.NotifyContext(ctx, os.Interrupt)
	defer cancel()

	// Connect to DB.
	db, err := sql.Open("sqlite3", cfg.DBPath)
	if err != nil {
		return fmt.Errorf("failed to open database: %w", err)
	}
	defer db.Close()

	// Run DB migrations.
	migrater, err := db_utils.NewMigrater(logger, db, cfg.DBMigrationDir)
	if err != nil {
		return fmt.Errorf("failed to create db migrater: %w", err)
	}
	logger.Info("Applying DB migrations.")
	migrater.Migrate(ctx)
	logger.Info("DB migrations complete")

	// Find all template files.
	templateFilenames, err := filepath.Glob("templates/*.html")
	if err != nil {
		return fmt.Errorf("failed to list template files: %w", err)
	}

	// Prepare TemplateExecutor.
	var tmpl utils.TemplateExecutor
	if cfg.Environment == utils.EEDevelopment {
		tmpl = utils.NewDevTemplate(templateFilenames...)
	} else {
		var err error
		tmpl, err = template.ParseFiles(templateFilenames...)
		if err != nil {
			return fmt.Errorf("failed to parse template files: %w", err)
		}
	}

	viteManifest, err := utils.LoadViteManifestFromFile(path.Join(cfg.ViteBuildDir, ".vite/manifest.json"))
	if err != nil {
		return fmt.Errorf("failed to load Vite manifest: %w", err)
	}

	http.Handle(
		"GET /hello/{name}",
		middleware.WithLogging(
			logger,
			handlers.BuildGetHelloHandler(tmpl, logger, cfg, "/"+viteManifest.MainJSFile.File, "/"+viteManifest.MainCSSFile.File),
		),
	)
	http.Handle(
		"GET /todo",
		middleware.WithLogging(
			logger,
			handlers.BuildGetTodoHandler(tmpl, logger, db, cfg, "/"+viteManifest.MainJSFile.File, "/"+viteManifest.MainCSSFile.File),
		),
	)
	http.Handle(
		"POST /todo",
		middleware.WithLogging(
			logger,
			handlers.BuildPostTodoHandler(tmpl, logger, db, cfg, "/"+viteManifest.MainJSFile.File, "/"+viteManifest.MainCSSFile.File),
		),
	)
	http.Handle(
		"DELETE /todo/{id}",
		middleware.WithLogging(
			logger,
			handlers.BuildDeleteTodoHandler(tmpl, logger, db, cfg, "/"+viteManifest.MainJSFile.File, "/"+viteManifest.MainCSSFile.File),
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
			http.StripPrefix("/assets/", http.FileServer(http.Dir(path.Join(cfg.ViteBuildDir, "assets")))),
		),
	)
	http.Handle(
		"GET /",
		middleware.WithLogging(
			logger,
			handlers.BuildGetIndexHandler(tmpl, cfg, logger, "/"+viteManifest.MainJSFile.File, "/"+viteManifest.MainCSSFile.File),
		),
	)

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

	cfg, err := utils.LoadAppConfigFromEnv()
	if err != nil {
		logger.Error(fmt.Sprintf("failed to load app config from env: %v", err))
		os.Exit(1)
	}
	logger.Info(fmt.Sprintf("Launching with app config:\n%v", cfg))

	if err := run(ctx, logger, cfg); err != nil {
		logger.Error(fmt.Sprintf("server error: %v", err))
		os.Exit(1)
	}
}
