# go-htmx-tailwind

A sample web app template built with:
- GO
- HTMX
- TailwindCSS

This sample project has an intentionally small dependency list. There are only 2 direct external Go dependencies: the sqlite driver and the `testify` testing library. The following features are all written from scratch:

- DB migrater
- Env variable loading
- Logging middleware


## Quick Start

Dev mode:
```bash
# Run vite in dev mode (from /frontend).
pnpm dev

# Build and run the app (from /).
ENVIRONMENT="development" go run ./cmd/app
```

Production mode:
```bash
# From /frontend, build the frontend bundle. The output will be written to frontend/dist/.
pnpm build

# Launch the GO server in production mode (from /).
ENVIRONMENT="release" go run ./cmd/app
```