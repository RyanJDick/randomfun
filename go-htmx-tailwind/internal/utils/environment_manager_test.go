package utils_test

import (
	"os"
	"path"
	"testing"

	"github.com/ryanjdick/go-htmx-tailwind/internal/utils"
	"github.com/stretchr/testify/assert"
)

func TestEnvironmentManager_LoadDotEnvFile(t *testing.T) {
	tmpDir := t.TempDir()

	// Create a test .env file.
	envFile := path.Join(tmpDir, ".env")
	dotEnvContents := []byte(
		//  A typical line.
		"ENV_VAR1=abc\n" +
			// A comment.
			"# This is a comment\n" +
			// A line with whitespace around the '='.
			"ENV_VAR2 = def\n" +
			// A blank line.
			"\n" +
			// A line with a quoted value (note the leading whitespace).
			"ENV_VAR3=\" ghi\"\n" +
			// A line with a quoted value and whitespace around the '='.
			"ENV_VAR4 = \" jkl\"\n" +
			// An key that will be superceded by an environment variable.
			"ENV_VAR5=mno\n")
	os.WriteFile(envFile, dotEnvContents, 0644)

	// Set env vars.
	t.Setenv("ENV_VAR6", "pqr")
	// Mask the .env ENV_VAR5 value with an environment variable.
	t.Setenv("ENV_VAR5", "stu")

	// Create a new EnvironmentManager and load the .env file.
	envManager := utils.NewEnvironmentManager()
	err := envManager.LoadDotEnvFile(envFile)
	assert.Nil(t, err)

	test_cases := []struct {
		key   string
		value string
	}{
		{"ENV_VAR1", "abc"},
		{"ENV_VAR2", "def"},
		{"ENV_VAR3", " ghi"},
		{"ENV_VAR4", " jkl"},
		{"ENV_VAR5", "stu"},
		{"ENV_VAR6", "pqr"},
	}

	for _, tc := range test_cases {
		t.Run(tc.key, func(t *testing.T) {
			value, ok := envManager.LookupEnv(tc.key)
			assert.True(t, ok)
			assert.Equal(t, tc.value, value)
		})
	}
}

func TestEnvironmentManager_LookupEnv(t *testing.T) {
	envManager := utils.NewEnvironmentManager()

	// Set an environment variable.
	t.Setenv("ENV_VAR1", "abc")

	// Lookup an environment variable that exists.
	value, ok := envManager.LookupEnv("ENV_VAR1")
	assert.True(t, ok)
	assert.Equal(t, "abc", value)

	// Lookup an environment variable that does not exist.
	_, ok = envManager.LookupEnv("ENV_VAR2")
	assert.False(t, ok)
}

func TestEnvironmentManager_GetEnvWithDefault(t *testing.T) {
	envManager := utils.NewEnvironmentManager()

	// Set an environment variable.
	t.Setenv("ENV_VAR1", "abc")

	// Get an environment variable that exists.
	value := envManager.GetEnvWithDefault("ENV_VAR1", "def")
	assert.Equal(t, "abc", value)

	// Get an environment variable that does not exist.
	value = envManager.GetEnvWithDefault("ENV_VAR2", "def")
	assert.Equal(t, "def", value)
}
