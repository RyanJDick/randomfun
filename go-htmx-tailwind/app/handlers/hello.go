package handlers

import (
	"html/template"
	"net/http"
	"time"
)

func getTime() string {
	return time.Now().Format(time.RFC1123)
}

func BuildGetHelloHandler(tmpl *template.Template) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		helloData := struct {
			Name string
			Time string
		}{
			Name: r.PathValue("name"),
			Time: getTime(),
		}
		tmpl.ExecuteTemplate(w, "hello.html", helloData)
	})
}

func BuildGetTimeHandler(tmpl *template.Template) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		timeData := struct{ Time string }{Time: getTime()}
		tmpl.ExecuteTemplate(w, "hello.time", timeData)
	})
}