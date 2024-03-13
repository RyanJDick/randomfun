package handlers

import (
	"log/slog"
	"net/http"

	"github.com/ryanjdick/go-htmx-tailwind/internal/template"
	"github.com/ryanjdick/go-htmx-tailwind/internal/utils"
)

type HelloTemplateData struct {
	Head template.HeadTemplateData
	Name string
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
		}
		err := tmpl.ExecuteTemplate(w, "hello.html", helloData)
		if err != nil {
			logger.Error("failed to render hello: %w", err)
			http.Error(w, "failed to render", http.StatusInternalServerError)
		}
	})
}
