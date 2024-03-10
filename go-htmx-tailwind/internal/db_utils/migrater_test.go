package db_utils

import (
	"context"
	"database/sql"
	"os"
	"path"
	"testing"

	_ "github.com/mattn/go-sqlite3"
	"github.com/stretchr/testify/assert"
)

// TestMigrater_Migrate tests the Migrater.Migrate method.
func TestMigrater_Migrate(t *testing.T) {
	tempDir := t.TempDir()

	// Populate tempDir with migration files.
	migrationDir := path.Join(tempDir, "migrations")
	err := os.Mkdir(migrationDir, 0755)
	assert.NoError(t, err)

	migration1 := []byte(`
		CREATE TABLE users (
			id INTEGER PRIMARY KEY,
			username TEXT NOT NULL,
			email TEXT NOT NULL
		);
	`)
	migration2 := []byte("ALTER TABLE users ADD COLUMN created_at TIMESTAMP;\n")
	migrationFiles := map[string][]byte{
		"0001_create_users_table.sql":   migration1,
		"0002_add_users_created_at.sql": migration2,
	}
	for fileName, fileContents := range migrationFiles {
		filePath := path.Join(migrationDir, fileName)
		err := os.WriteFile(filePath, fileContents, 0644)
		assert.NoError(t, err)
	}

	// Open the database.
	dbPath := path.Join(tempDir, "test.sqlite")
	db, err := sql.Open("sqlite3", dbPath)
	assert.NoError(t, err)
	defer db.Close()

	// Perform the database migration.
	migrater, err := NewMigrater(db, migrationDir)
	assert.NoError(t, err)
	ctx := context.Background()
	err = migrater.Migrate(ctx)
	assert.NoError(t, err)

	// Verify the migrations table.
	completedMigrations, err := migrater.getAllMigrations(ctx)
	assert.NoError(t, err)
	assert.Len(t, completedMigrations, 2)
	assert.Equal(t, 1, completedMigrations[0].Version)
	assert.Equal(t, "0001_create_users_table", completedMigrations[0].Name)
	assert.Equal(t, 2, completedMigrations[1].Version)
	assert.Equal(t, "0002_add_users_created_at", completedMigrations[1].Name)

	// Migrations should be idempotent.
	err = migrater.Migrate(ctx)
	assert.NoError(t, err)
	completedMigrations, err = migrater.getAllMigrations(ctx)
	assert.NoError(t, err)
	assert.Len(t, completedMigrations, 2)
}

// TestMigrater_NoMigrations tests the Migrater.Migrate method when there are no migrations to apply.
func TestMigrater_NoMigrations(t *testing.T) {
	tempDir := t.TempDir()
	migrationDir := path.Join(tempDir, "migrations")

	// Open the database.
	dbPath := path.Join(tempDir, "test.sqlite")
	db, err := sql.Open("sqlite3", dbPath)
	assert.NoError(t, err)
	defer db.Close()

	// Perform the database migration.
	migrater, err := NewMigrater(db, migrationDir)
	assert.NoError(t, err)
	ctx := context.Background()
	err = migrater.Migrate(ctx)
	assert.NoError(t, err)

	// Verify the migrations table.
	completedMigrations, err := migrater.getAllMigrations(ctx)
	assert.NoError(t, err)
	assert.Len(t, completedMigrations, 0)
}

// TestMigrater_MigratePartial tests the Migrater.Migrate method when some migrations have already been applied.
func TestMigrater_MigratePartial(t *testing.T) {
	tempDir := t.TempDir()

	// Populate tempDir with migration files.
	migrationDir := path.Join(tempDir, "migrations")
	err := os.Mkdir(migrationDir, 0755)
	assert.NoError(t, err)

	migration1 := []byte(`
		CREATE TABLE users (
			id INTEGER PRIMARY KEY,
			username TEXT NOT NULL,
			email TEXT NOT NULL
		);
	`)
	migrationFiles := map[string][]byte{
		"0001_create_users_table.sql": migration1,
	}
	for fileName, fileContents := range migrationFiles {
		filePath := path.Join(migrationDir, fileName)
		err := os.WriteFile(filePath, fileContents, 0644)
		assert.NoError(t, err)
	}

	// Open the database.
	dbPath := path.Join(tempDir, "test.sqlite")
	db, err := sql.Open("sqlite3", dbPath)
	assert.NoError(t, err)
	defer db.Close()

	// Perform the database migration.
	migrater, err := NewMigrater(db, migrationDir)
	assert.NoError(t, err)
	ctx := context.Background()
	err = migrater.Migrate(ctx)
	assert.NoError(t, err)

	// Verify the migrations table.
	completedMigrations, err := migrater.getAllMigrations(ctx)
	assert.NoError(t, err)
	assert.Len(t, completedMigrations, 1)
	assert.Equal(t, 1, completedMigrations[0].Version)
	assert.Equal(t, "0001_create_users_table", completedMigrations[0].Name)

	// Add a new migration file.
	migration2 := []byte("ALTER TABLE users ADD COLUMN created_at TIMESTAMP;\n")
	migrationFiles = map[string][]byte{
		"0001_create_users_table.sql":   migration1,
		"0002_add_users_created_at.sql": migration2,
	}
	for fileName, fileContents := range migrationFiles {
		filePath := path.Join(migrationDir, fileName)
		err := os.WriteFile(filePath, fileContents, 0644)
		assert.NoError(t, err)
	}

	// Perform the database migration.
	migrater, err = NewMigrater(db, migrationDir)
	assert.NoError(t, err)
	err = migrater.Migrate(ctx)
	assert.NoError(t, err)

	// Verify the migrations table.
	completedMigrations, err = migrater.getAllMigrations(ctx)
	assert.NoError(t, err)
	assert.Len(t, completedMigrations, 2)
	assert.Equal(t, 1, completedMigrations[0].Version)
	assert.Equal(t, "0001_create_users_table", completedMigrations[0].Name)
	assert.Equal(t, 2, completedMigrations[1].Version)
	assert.Equal(t, "0002_add_users_created_at", completedMigrations[1].Name)
}

// TestMigrater_MigrationMismatch tests the Migrater.Migrate method when the migration files do not match the database.
func TestMigrater_MigrationMismatch(t *testing.T) {
	tempDir := t.TempDir()

	// Populate tempDir with migration files.
	migrationDir := path.Join(tempDir, "migrations")
	err := os.Mkdir(migrationDir, 0755)
	assert.NoError(t, err)

	migration1 := []byte(`
		CREATE TABLE users (
			id INTEGER PRIMARY KEY,
			username TEXT NOT NULL,
			email TEXT NOT NULL
		);
	`)
	migrationFiles := map[string][]byte{
		"0001_create_users_table.sql": migration1,
	}
	for fileName, fileContents := range migrationFiles {
		filePath := path.Join(migrationDir, fileName)
		err := os.WriteFile(filePath, fileContents, 0644)
		assert.NoError(t, err)
	}

	// Open the database.
	dbPath := path.Join(tempDir, "test.sqlite")
	db, err := sql.Open("sqlite3", dbPath)
	assert.NoError(t, err)
	defer db.Close()

	// Perform the database migration.
	migrater, err := NewMigrater(db, migrationDir)
	assert.NoError(t, err)
	ctx := context.Background()
	err = migrater.Migrate(ctx)
	assert.NoError(t, err)

	// Verify the migrations table.
	completedMigrations, err := migrater.getAllMigrations(ctx)
	assert.NoError(t, err)
	assert.Len(t, completedMigrations, 1)
	assert.Equal(t, 1, completedMigrations[0].Version)
	assert.Equal(t, "0001_create_users_table", completedMigrations[0].Name)

	// Update the migrations directory with a different migration.
	migrationDir = path.Join(tempDir, "migrations2")
	err = os.Mkdir(migrationDir, 0755)
	assert.NoError(t, err)
	migration2 := []byte("ALTER TABLE users ADD COLUMN created_at TIMESTAMP;\n")
	migrationFiles = map[string][]byte{
		"0002_add_users_created_at.sql": migration2,
	}
	for fileName, fileContents := range migrationFiles {
		filePath := path.Join(migrationDir, fileName)
		err := os.WriteFile(filePath, fileContents, 0644)
		assert.NoError(t, err)
	}

	// Attempting to run a migration should return an error.
	migrater, err = NewMigrater(db, migrationDir)
	assert.NoError(t, err)
	err = migrater.Migrate(ctx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "the DB migration history does not match the migration files")
}
