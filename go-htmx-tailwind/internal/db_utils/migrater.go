package db_utils

import (
	"context"
	"database/sql"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
)

type migration struct {
	Version int
	Name    string
}

type Migrater struct {
	logger         *slog.Logger
	db             *sql.DB
	migrationFiles []*migrationFile
}

func NewMigrater(logger *slog.Logger, db *sql.DB, migrationDir string) (*Migrater, error) {
	// Get all migration files from the migration directory.
	files, err := filepath.Glob(filepath.Join(migrationDir, "*.sql"))
	if err != nil {
		return nil, fmt.Errorf("failed to find migration files: %w", err)
	}

	// Convert []string to []*migrationFile.
	var migrationFiles []*migrationFile
	for _, file := range files {

		migrationFile, err := newMigrationFile(file)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize migration file: %w", err)
		}
		migrationFiles = append(migrationFiles, migrationFile)
	}

	// Sort migration files by leading integer.
	sort.Slice(migrationFiles, func(i, j int) bool {
		return migrationFiles[i].Version() < migrationFiles[j].Version()
	})

	return &Migrater{logger: logger, db: db, migrationFiles: migrationFiles}, nil
}

// getAllMigrations returns all migrations from the migrations table ordered by ascending version.
func (m *Migrater) getAllMigrations(ctx context.Context) ([]*migration, error) {
	rows, err := m.db.QueryContext(ctx, "SELECT version, name FROM migrations ORDER BY version ASC;")
	if err != nil {
		return nil, fmt.Errorf("failed to query migrations: %w", err)
	}
	defer rows.Close()

	var migrations []*migration
	for rows.Next() {
		var m migration
		if err := rows.Scan(&m.Version, &m.Name); err != nil {
			return nil, fmt.Errorf("failed to scan migration rows: %w", err)
		}
		migrations = append(migrations, &m)
	}

	return migrations, nil
}

func (m *Migrater) Migrate(ctx context.Context) error {
	// Create the migrations table if it does not exist.
	_, err := m.db.ExecContext(ctx, `
		CREATE TABLE IF NOT EXISTS migrations (
			id INTEGER PRIMARY KEY,
			version INTEGER NOT NULL UNIQUE,
			name TEXT NOT NULL,
			created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
		);
	`)
	if err != nil {
		return fmt.Errorf("failed to create migrations table: %w", err)
	}

	// Get the current migration history.
	completedMigrations, err := m.getAllMigrations(ctx)
	if err != nil {
		return fmt.Errorf("failed to get completed migrations: %w", err)
	}

	// Verify that the DB migration history matches the migration files.
	for migrationIndex, completedMigration := range completedMigrations {
		if migrationIndex >= len(m.migrationFiles) {
			return fmt.Errorf("the DB migration history contains more versions than migration files")
		}
		if completedMigration.Version != m.migrationFiles[migrationIndex].Version() {
			return fmt.Errorf(
				"the DB migration history does not match the migration files (DB version %d vs. expected %d)",
				completedMigration.Version,
				m.migrationFiles[migrationIndex].Version(),
			)
		}
	}

	// Log number of migrations to be executed.
	m.logger.Info(fmt.Sprintf("Applying %d new database migration(s).", len(m.migrationFiles)-len(completedMigrations)))

	// Migrate the database to the latest version.
	for _, migrationFile := range m.migrationFiles[len(completedMigrations):] {
		// Read the migration file.
		migrationSQL, err := os.ReadFile(migrationFile.FilePath())
		if err != nil {
			return fmt.Errorf("failed to read migration file: %w", err)
		}

		// Start a transaction.
		tx, err := m.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to start db transaction: %w", err)
		}
		defer tx.Rollback()

		// Execute the migration.
		if _, err := tx.ExecContext(ctx, string(migrationSQL)); err != nil {
			return fmt.Errorf(
				"failed to execute query for migration '%s': %w",
				migrationFile.FilePath(),
				err,
			)
		}

		// Record the migration in the migrations table.
		_, err = tx.ExecContext(
			ctx,
			"INSERT INTO migrations (version, name) VALUES ($1, $2);",
			migrationFile.Version(),
			migrationFile.Name(),
		)
		if err != nil {
			return fmt.Errorf("failed to record migration in migrations table: %w", err)
		}

		// Commit the transaction.
		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit db transaction: %w", err)
		}

		// Log that a migration was applied.
		m.logger.Info(fmt.Sprintf("Applied migration %d: '%s'.", migrationFile.Version(), migrationFile.Name()))
	}

	// Get updated migration history, and log the current version.
	completedMigrations, err = m.getAllMigrations(ctx)
	if err != nil {
		return fmt.Errorf("failed to get completed migrations: %w", err)
	}
	if len(completedMigrations) == 0 {
		m.logger.Info("No database migrations have been applied yet.")
	} else {
		m.logger.Info(fmt.Sprintf("The current database version is %d.", completedMigrations[len(completedMigrations)-1].Version))
	}

	return nil
}
