package utils

import (
	"encoding/json"
	"fmt"
	"os"
)

type viteManifestFileInfo struct {
	File    string `json:"file"`
	Src     string `json:"src"`
	IsEntry bool   `json:"isEntry"`
}

// ViteManifest is used to load '.vite/manifest.json' files.
type ViteManifest struct {
	MainJSFile  viteManifestFileInfo `json:"src/main.ts"`
	MainCSSFile viteManifestFileInfo `json:"src/main.css"`
}

func LoadViteManifestFromFile(filename string) (ViteManifest, error) {
	viteManifest := ViteManifest{}

	manifestBytes, err := os.ReadFile(filename)
	if err != nil {
		return viteManifest, fmt.Errorf("failed to read file '%s': %w", filename, err)
	}

	if err = json.Unmarshal(manifestBytes, &viteManifest); err != nil {
		return viteManifest, fmt.Errorf("failed to unmarshal json from '%s': %w", filename, err)
	}

	return viteManifest, err
}
