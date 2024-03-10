package db_utils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestMigrationFile_NewMigrationFile(t *testing.T) {
	testCases := []struct {
		testName        string
		filePath        string
		expectedName    string
		expectedVersion int
		expectError     bool
	}{
		{
			testName:        "valid file path",
			filePath:        "0001_initial.sql",
			expectedName:    "0001_initial",
			expectedVersion: 1,
			expectError:     false,
		},
		{
			testName:        "valid file path with directory",
			filePath:        "path/to/0001_initial.sql",
			expectedName:    "0001_initial",
			expectedVersion: 1,
			expectError:     false,
		},
		{
			testName:    "invalid file extension",
			filePath:    "0001_initial.txt",
			expectError: true,
		},
		{
			testName:    "file name does not start with version number",
			filePath:    "x_0001_initial.sql",
			expectError: true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			mf, err := newMigrationFile(tc.filePath)
			if tc.expectError {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tc.expectedName, mf.Name())
				assert.Equal(t, tc.expectedVersion, mf.Version())
			}
		})
	}
}
