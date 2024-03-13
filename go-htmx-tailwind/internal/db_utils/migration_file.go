package db_utils

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strconv"
)

type migrationFile struct {
	filePath string
	name     string
	version  int
}

func (m *migrationFile) FilePath() string { return m.filePath }
func (m *migrationFile) Name() string     { return m.name }
func (m *migrationFile) Version() int     { return m.version }

var integerRegex = regexp.MustCompile(`^\d+`)

func newMigrationFile(filePath string) (*migrationFile, error) {
	_, fileName := filepath.Split(filePath)
	ext := filepath.Ext(fileName)
	if ext != ".sql" {
		return nil, fmt.Errorf("file %s is not a .sql file", filePath)
	}

	name := fileName[:len(fileName)-len(ext)]

	versionString := integerRegex.FindString(fileName)
	version, err := strconv.Atoi(versionString)
	if err != nil {
		return nil, fmt.Errorf("failed to parse version from migration file: %w", err)
	}

	return &migrationFile{
		filePath: filePath,
		name:     name,
		version:  version,
	}, nil
}
