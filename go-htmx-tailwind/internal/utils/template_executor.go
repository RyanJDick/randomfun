package utils

import (
	"fmt"
	"html/template"
	"io"
)

// TemplateExecutor is a thin interface implemented by both DevTemplateExecutor
// and ProdTemplateExecutor.
type TemplateExecutor interface {
	ExecuteTemplate(wr io.Writer, name string, data interface{}) error
}

// DevTemplate is a TemplateExecutor that reloads templates on every request.
// This is useful during development.
type DevTemplate struct {
	filenames []string
}

// NewDevTemplate constructs a neww DevTemplate.
func NewDevTemplate(filenames ...string) *DevTemplate {
	return &DevTemplate{filenames: filenames}
}

func (dt *DevTemplate) ExecuteTemplate(wr io.Writer, name string, data interface{}) error {
	tmpl, err := template.ParseFiles(dt.filenames...)
	if err != nil {
		return fmt.Errorf("failed to parse template files: %w", err)
	}

	if err := tmpl.ExecuteTemplate(wr, name, data); err != nil {
		return fmt.Errorf("failed to execute template '%s': %w", name, err)
	}

	return nil
}
