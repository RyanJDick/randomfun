package main

import (
	"encoding/json"
	"html/template"
	"log"
	"net/http"

	"github.com/ryanjdick/go-htmx-tailwind/app/handlers"
)

func main() {
	// TODO: Log every request, including timing, and response code.
	// TODO: Log all available routes on startup.

	tmpl := template.Must(template.ParseFiles("templates/hello.html"))

	http.HandleFunc("GET /ping", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"message": "pong"})
	})

	http.Handle("GET /hello/{name}", handlers.BuildGetHelloHandler(tmpl))
	http.Handle("GET /time", handlers.BuildGetTimeHandler(tmpl))
	http.Handle("GET /", http.FileServer(http.Dir("static")))

	// logger := slog.Default()

	log.Fatal(http.ListenAndServe(":8080", nil))
}
