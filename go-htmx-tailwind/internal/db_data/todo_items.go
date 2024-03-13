package dbdata

import (
	"context"
	"database/sql"
	"fmt"
)

type TodoItem struct {
	Id          int
	Description string
	CreatedAt   string
}

// GetAllTodoItems returns all todo items from the todo_items table.
func GetAllTodoItems(ctx context.Context, db *sql.DB) ([]TodoItem, error) {
	rows, err := db.QueryContext(ctx, "SELECT id, description, created_at FROM todo_items ORDER BY created_at DESC;")
	if err != nil {
		return nil, fmt.Errorf("failed to query todo items: %w", err)
	}
	defer rows.Close()

	var todoItems []TodoItem
	for rows.Next() {
		var ti TodoItem
		err := rows.Scan(&ti.Id, &ti.Description, &ti.CreatedAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan todo item: %w", err)
		}
		todoItems = append(todoItems, ti)
	}
	return todoItems, nil
}

// AddTodoItem adds a new todo item to the todo_items table.
func AddTodoItem(ctx context.Context, db *sql.DB, description string) error {
	_, err := db.ExecContext(ctx, "INSERT INTO todo_items (description) VALUES (?)", description)
	if err != nil {
		return fmt.Errorf("failed to insert todo item: %w", err)
	}
	return nil
}

// DeleteTodoItem deletes a todo item from the todo_items table.
func DeleteTodoItem(ctx context.Context, db *sql.DB, id int) error {
	_, err := db.ExecContext(ctx, "DELETE FROM todo_items WHERE id = ?", id)
	if err != nil {
		return fmt.Errorf("failed to delete todo item: %w", err)
	}
	return nil
}
