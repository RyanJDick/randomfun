package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/ryanjdick/go-htmx-tailwind/app"
)

func handleIndex(w http.ResponseWriter, r *http.Request) {
	name, err := app.GetRouteParam(r, 0)
	if err != nil {
		http.NotFound(w, r)
		return
	}

	fmt.Fprintf(w, "Hi %s!", name)
}

func main() {

	rh := new(app.RouteHandler)

	rh.GetFn(`^/hi/([a-zA-Z0-9]+)/?$`, handleIndex)

	http.Handle("/", rh)

	// http.HandleFunc("/", handleIndex)

	// fs := http.FileServer(http.Dir("static/"))
	// http.Handle("/static/", http.StripPrefix("/static/", fs))

	log.Fatal(http.ListenAndServe(":8080", nil))
}
