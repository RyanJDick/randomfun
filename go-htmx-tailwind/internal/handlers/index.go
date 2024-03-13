package handlers

import (
	"fmt"
	"log/slog"
	"net/http"

	"github.com/ryanjdick/go-htmx-tailwind/internal/template"
	"github.com/ryanjdick/go-htmx-tailwind/internal/utils"
)

type IndexTemplateData struct {
	Head template.HeadTemplateData
}

func BuildGetIndexHandler(tmpl utils.TemplateExecutor, envConfig *utils.AppConfig, logger *slog.Logger, mainJSPath string, mainCSSPath string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		indexData := IndexTemplateData{
			Head: template.HeadTemplateData{
				Title:         "go-htmx-tailwind",
				IsDevelopment: envConfig.Environment == utils.EEDevelopment,
				MainJSPath:    mainJSPath,
				MainCSSPath:   mainCSSPath,
			},
		}
		err := tmpl.ExecuteTemplate(w, "index.html", indexData)
		if err != nil {
			logger.Error(fmt.Sprintf("failed to render index: %w", err))
			http.Error(w, "failed to render", http.StatusInternalServerError)
		}
	})
}
