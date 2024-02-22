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

type EnvConfig struct {
	Environment EnvironmentEnum
}

func LoadEnv() (*EnvConfig, error) {
	environment, err := NewEnvironmentEnum(os.Getenv("ENVIRONMENT"))
	if err != nil {
		return nil, fmt.Errorf("failed to create environment value: %w", err)
	}

	return &EnvConfig{Environment: environment}, nil
}
