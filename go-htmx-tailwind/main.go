package main

import (
	"html/template"
	"log"
	"log/slog"
	"net/http"

	"github.com/ryanjdick/go-htmx-tailwind/app/handlers"
	"github.com/ryanjdick/go-htmx-tailwind/app/middleware"
)

func main() {
	// TODO: Log every request, including timing, and response code.
	// TODO: Log all available routes on startup.

	tmpl := template.Must(template.ParseFiles("templates/hello.html"))

	logger := slog.Default()

	http.Handle("GET /hello/{name}", middleware.WithLogging(logger, handlers.BuildGetHelloHandler(tmpl)))
	http.Handle("GET /time", middleware.WithLogging(logger, handlers.BuildGetTimeHandler(tmpl)))
	http.Handle("GET /", middleware.WithLogging(logger, http.FileServer(http.Dir("static"))))

	log.Fatal(http.ListenAndServe(":8080", nil))
}
