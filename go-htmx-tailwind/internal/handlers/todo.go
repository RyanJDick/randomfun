package handlers

import (
	"database/sql"
	"fmt"
	"log/slog"
	"net/http"
	"strconv"

	dbdata "github.com/ryanjdick/go-htmx-tailwind/internal/db_data"
	"github.com/ryanjdick/go-htmx-tailwind/internal/template"
	"github.com/ryanjdick/go-htmx-tailwind/internal/utils"
)

type TodoTemplateData struct {
	Head      template.HeadTemplateData
	TodoItems []dbdata.TodoItem
}

func BuildGetTodoHandler(
	tmpl utils.TemplateExecutor,
	logger *slog.Logger,
	db *sql.DB,
	envConfig *utils.AppConfig,
	mainJSPath string,
	mainCSSPath string,
) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {

		// Get the todo items from the database.
		todoItems, err := dbdata.GetAllTodoItems(r.Context(), db)
		if err != nil {
			logger.Error("failed to get todo items: %w", err)
			http.Error(w, "failed to get todo items", http.StatusInternalServerError)
			return
		}

		todoData := TodoTemplateData{
			Head: template.HeadTemplateData{
				Title:         "TODO",
				IsDevelopment: envConfig.Environment == utils.EEDevelopment,
				MainJSPath:    mainJSPath,
				MainCSSPath:   mainCSSPath,
			},
			TodoItems: todoItems,
		}
		err = tmpl.ExecuteTemplate(w, "todo.html", todoData)
		if err != nil {
			logger.Error("failed to render /todo: %w", err)
			http.Error(w, "failed to render", http.StatusInternalServerError)
		}
	})
}

func BuildPostTodoHandler(
	tmpl utils.TemplateExecutor,
	logger *slog.Logger,
	db *sql.DB,
	envConfig *utils.AppConfig,
	mainJSPath string,
	mainCSSPath string,
) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Add the new todo item to the database.
		description := r.FormValue("description")
		if len(description) > 0 {
			logger.Info(fmt.Sprintf("Adding todo item: %s", description))
			err := dbdata.AddTodoItem(r.Context(), db, description)
			if err != nil {
				logger.Error(fmt.Sprintf("failed to add todo item: %v", err))
				http.Error(w, "failed to add todo item", http.StatusInternalServerError)
				return
			}
		} else {
			logger.Info("Empty todo item description. Not adding to the database.")
		}

		// Get the todo items from the database.
		todoItems, err := dbdata.GetAllTodoItems(r.Context(), db)
		if err != nil {
			logger.Error("failed to get todo items: %w", err)
			http.Error(w, "failed to get todo items", http.StatusInternalServerError)
			return
		}

		todoData := TodoTemplateData{
			Head: template.HeadTemplateData{
				Title:         "TODO",
				IsDevelopment: envConfig.Environment == utils.EEDevelopment,
				MainJSPath:    mainJSPath,
				MainCSSPath:   mainCSSPath,
			},
			TodoItems: todoItems,
		}

		// Render the updated todo list.
		err = tmpl.ExecuteTemplate(w, "todo.list", todoData)
		if err != nil {
			logger.Error("failed to render todo.list: %w", err)
			http.Error(w, "failed to render", http.StatusInternalServerError)
		}
	})
}

func BuildDeleteTodoHandler(
	tmpl utils.TemplateExecutor,
	logger *slog.Logger,
	db *sql.DB,
	envConfig *utils.AppConfig,
	mainJSPath string,
	mainCSSPath string,
) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Delete the todo item from the database.
		idStr := r.PathValue("id")
		id, err := strconv.Atoi(idStr)
		if err != nil {
			logger.Error(fmt.Sprintf("failed to convert id to int: %v", err))
			http.Error(w, "failed to convert id to int", http.StatusBadRequest)
			return
		}
		logger.Info(fmt.Sprintf("Deleting todo item: %d", id))
		err = dbdata.DeleteTodoItem(r.Context(), db, id)
		if err != nil {
			logger.Error(fmt.Sprintf("failed to delete todo item: %v", err))
			http.Error(w, "failed to delete todo item", http.StatusInternalServerError)
			return
		}

		// Get the todo items from the database.
		todoItems, err := dbdata.GetAllTodoItems(r.Context(), db)
		if err != nil {
			logger.Error("failed to get todo items: %w", err)
			http.Error(w, "failed to get todo items", http.StatusInternalServerError)
			return
		}

		todoData := TodoTemplateData{
			Head: template.HeadTemplateData{
				Title:         "TODO",
				IsDevelopment: envConfig.Environment == utils.EEDevelopment,
				MainJSPath:    mainJSPath,
				MainCSSPath:   mainCSSPath,
			},
			TodoItems: todoItems,
		}

		// Render the updated todo list.
		err = tmpl.ExecuteTemplate(w, "todo.list", todoData)
		if err != nil {
			logger.Error("failed to render todo.list: %w", err)
			http.Error(w, "failed to render", http.StatusInternalServerError)
		}
	})
}
