package main

import (
	"log"
	"log/slog"
	"net/http"

	"github.com/ryanjdick/go-htmx-tailwind/app"
	"github.com/ryanjdick/go-htmx-tailwind/app/handlers"
)

func main() {

	logger := slog.Default()

	rh := new(app.RouteHandler)

	helloHandler := &handlers.HelloHandler{Logger: logger}

	rh.Get(`^/hello/([a-zA-Z0-9]+)/?$`, helloHandler)

	http.Handle("/", rh)

	// http.HandleFunc("/", handleIndex)

	// fs := http.FileServer(http.Dir("static/"))
	// http.Handle("/static/", http.StripPrefix("/static/", fs))

	log.Fatal(http.ListenAndServe(":8080", nil))
}
