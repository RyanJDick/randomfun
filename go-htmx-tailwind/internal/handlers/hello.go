package handlers

import (
	"log/slog"
	"net/http"
	"time"

	"github.com/ryanjdick/go-htmx-tailwind/internal/template"
	"github.com/ryanjdick/go-htmx-tailwind/internal/utils"
)

func getTime() string {
	return time.Now().Format(time.RFC1123)
}

type HelloTemplateData struct {
	Head template.HeadTemplateData
	Name string
	Time string
}

func BuildGetHelloHandler(tmpl utils.TemplateExecutor, logger *slog.Logger, envConfig *utils.AppConfig, mainJSPath string, mainCSSPath string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		helloData := HelloTemplateData{
			Head: template.HeadTemplateData{
				Title:         "Hello",
				IsDevelopment: envConfig.Environment == utils.EEDevelopment,
				MainJSPath:    mainJSPath,
				MainCSSPath:   mainCSSPath,
			},
			Name: r.PathValue("name"),
			Time: getTime(),
		}
		err := tmpl.ExecuteTemplate(w, "hello.html", helloData)
		if err != nil {
			logger.Error("failed to render hello: %w", err)
			http.Error(w, "failed to render", http.StatusInternalServerError)
		}
	})
}

func BuildGetTimeHandler(tmpl utils.TemplateExecutor) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		timeData := struct{ Time string }{Time: getTime()}
		tmpl.ExecuteTemplate(w, "hello.time", timeData)
	})
}
