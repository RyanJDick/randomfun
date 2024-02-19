package middleware

import (
	"fmt"
	"log/slog"
	"net/http"
	"time"
)

// WithLogging is a handler middleware that logs the duration of each request.
func WithLogging(logger *slog.Logger, handler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		handler.ServeHTTP(w, r)
		elapsed := time.Since(start)
		logger.Info(fmt.Sprintf("%v %v in %v", r.Method, r.URL.Path, elapsed))
	})
}
