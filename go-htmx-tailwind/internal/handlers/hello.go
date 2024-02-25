package handlers

import (
	"net/http"
	"time"

	"github.com/ryanjdick/go-htmx-tailwind/internal/utils"
)

func getTime() string {
	return time.Now().Format(time.RFC1123)
}

func BuildGetHelloHandler(tmpl utils.TemplateExecutor, envConfig *utils.AppConfig, mainJSPath string, mainCSSPath string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		helloData := struct {
			IsDevelopment bool
			MainJSPath    string
			MainCSSPath   string
			Name          string
			Time          string
		}{
			IsDevelopment: envConfig.Environment == utils.EEDevelopment,
			MainJSPath:    mainJSPath,
			MainCSSPath:   mainCSSPath,
			Name:          r.PathValue("name"),
			Time:          getTime(),
		}
		tmpl.ExecuteTemplate(w, "hello.html", helloData)
	})
}

func BuildGetTimeHandler(tmpl utils.TemplateExecutor) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		timeData := struct{ Time string }{Time: getTime()}
		tmpl.ExecuteTemplate(w, "hello.time", timeData)
	})
}