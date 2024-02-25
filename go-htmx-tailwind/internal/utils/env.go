package utils

import (
	"fmt"
	"os"
)

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
}

func (a *AppConfig) String() string {
	return fmt.Sprintf(
		"Host: %s\nPort: %s\nViteBuildDir: %s\nEnvironment: %s",
		a.Host, a.Port, a.ViteBuildDir, a.Environment.environment,
	)
}

func LoadAppConfigFromEnv() (*AppConfig, error) {
	environment, err := NewEnvironmentEnum(os.Getenv("ENVIRONMENT"))
	if err != nil {
		return nil, fmt.Errorf("failed to create environment value: %w", err)
	}

	return &AppConfig{
		Host:         "localhost",
		Port:         "8080",
		ViteBuildDir: "frontend/dist",
		Environment:  environment,
	}, nil
}
