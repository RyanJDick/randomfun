package utils

import "fmt"

type EnvironmentEnum struct {
	environment string
}

func NewEnvironmentEnum(s string) (EnvironmentEnum, error) {
	switch s {
	case EEDevelopment.environment:
		return EEDevelopment, nil
	case EERelease.environment:
		return EERelease, nil
	default:
		return EEUnknown, fmt.Errorf("'%s' is not a supported environment type", s)
	}
}

var (
	EEUnknown     = EnvironmentEnum{"unknown"}
	EEDevelopment = EnvironmentEnum{"development"}
	EERelease     = EnvironmentEnum{"release"}
)

type AppConfig struct {
	Host         string
	Port         string
	ViteBuildDir string
	Environment  EnvironmentEnum

	// DB configs
	DBPath         string
	DBMigrationDir string
}

func (a *AppConfig) String() string {
	return fmt.Sprintf(
		"Host: %s\nPort: %s\nViteBuildDir: %s\nEnvironment: %s",
		a.Host, a.Port, a.ViteBuildDir, a.Environment.environment,
	)
}

// LoadAppConfigFromEnv loads the application configuration from environment
// variables or a .env file. Environment variables take precedence over values
// set in the .env.
func LoadAppConfigFromEnv() (*AppConfig, error) {
	envManager := NewEnvironmentManager()
	envManager.LoadDotEnvFile(".env")

	// Load "ENVIRONMENT" environment variable.
	environmentValue, ok := envManager.LookupEnv("ENVIRONMENT")
	if !ok {
		return nil, fmt.Errorf("environment variable 'ENVIRONMENT' is not set")
	}
	environment, err := NewEnvironmentEnum(environmentValue)
	if err != nil {
		return nil, fmt.Errorf("failed to create environment value: %w", err)
	}

	return &AppConfig{
		Host:           envManager.GetEnvWithDefault("HOST", "localhost"),
		Port:           envManager.GetEnvWithDefault("PORT", "8080"),
		ViteBuildDir:   envManager.GetEnvWithDefault("VITE_BUILD_DIR", "frontend/dist"),
		Environment:    environment,
		DBPath:         envManager.GetEnvWithDefault("DB_PATH", "db.sqlite"),
		DBMigrationDir: envManager.GetEnvWithDefault("DB_MIGRATION_DIR", "migrations"),
	}, nil
}
