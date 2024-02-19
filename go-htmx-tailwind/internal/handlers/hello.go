package handlers

import (
	"html/template"
	"net/http"
	"time"
)

func getTime() string {
	return time.Now().Format(time.RFC1123)
}

func BuildGetHelloHandler(tmpl *template.Template, mainJSPath string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		helloData := struct {
			MainJSPath string
			Name       string
			Time       string
		}{
			MainJSPath: mainJSPath,
			Name:       r.PathValue("name"),
			Time:       getTime(),
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
