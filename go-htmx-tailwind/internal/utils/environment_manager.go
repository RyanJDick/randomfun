package utils

import (
	"fmt"
	"os"
	"regexp"
	"strings"
)

// EvironmentManager is a utility for loading values from environment variables
// and .env files.
type EnvironmentManager struct {
	dotEnvMap map[string]string
}

func NewEnvironmentManager() *EnvironmentManager {
	return &EnvironmentManager{
		dotEnvMap: make(map[string]string),
	}
}

var dotEnvLineRegex = regexp.MustCompile(`^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)\s*$`)

// loadDotEnv parses the contents of a .env file and stores the key-value pairs.
// New key-value pairs will overwrite existing ones with the same key.
func (e *EnvironmentManager) loadDotEnv(bytes []byte) {
	dotEnvString := string(bytes)

	for _, dotEnvLine := range strings.Split(dotEnvString, "\n") {
		submatches := dotEnvLineRegex.FindStringSubmatch(dotEnvLine)
		if len(submatches) != 3 {
			// Valid lines have 3 submatches: the entire line, the key, and the value.
			continue
		}
		key := strings.TrimSpace(submatches[1])
		value := strings.TrimSpace(submatches[2])

		// Remove surrounding quotes from value.
		if value[0] == '"' && value[len(value)-1] == '"' {
			value = value[1 : len(value)-1]
		}
		e.dotEnvMap[key] = value
	}
}

// LoadDotEnvFile reads a .env file and loads the key-value pairs into the EnvironmentManager.
// New key-value pairs will overwrite existing ones with the same key.
func (e *EnvironmentManager) LoadDotEnvFile(filepath string) error {
	dotEnvBytes, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("failed to read .env file: %w", err)
	}

	e.loadDotEnv(dotEnvBytes)
	return nil
}

// LookupEnv retrieves the value of the variable with the given key.
// Environment variables have the highest precedence, followed by values loaded
// from a .env file.
// If the key is found, then the value (which may be empty) is returned, and the
// boolean return value is true.
// If the key is not found, an empty string is returned and the boolean return
// value is false.
func (e *EnvironmentManager) LookupEnv(key string) (string, bool) {
	if value, ok := os.LookupEnv(key); ok {
		return value, true
	}
	if value, ok := e.dotEnvMap[key]; ok {
		return value, true
	}
	return "", false
}

// GetEnvWithDefault retrieves the value of the variable with the given key.
// The precedence of possible return values is:
// 1. Environment variable
// 2. Value loaded from a .env file
// 3. Default value (`def`)
func (e *EnvironmentManager) GetEnvWithDefault(key string, def string) string {
	value, ok := e.LookupEnv(key)
	if !ok {
		return def
	}
	return value
}
