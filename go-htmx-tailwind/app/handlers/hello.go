package handlers

import (
	"fmt"
	"html/template"
	"log/slog"
	"net/http"

	"github.com/ryanjdick/go-htmx-tailwind/app"
)

var templates = template.Must(template.ParseFiles("templates/hello.html"))

type helloData struct {
	Name string
}

type HelloHandler struct {
	Logger *slog.Logger
}

func (hh *HelloHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	name, err := app.GetRouteParam(r, 0)
	if err != nil {
		hh.Logger.Warn(fmt.Sprintf("failed to get route param: %v", err.Error()))
		http.NotFound(w, r)
		return
	}

	data := helloData{Name: name}
	err = templates.ExecuteTemplate(w, "hello.html", data)
	if err != nil {
		hh.Logger.Warn(fmt.Sprintf("failed to execute hello template: %v", err.Error()))
		http.Error(w, "Internal server error.", http.StatusInternalServerError)
	}
}
