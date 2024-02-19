import { defineConfig} from "vite"

export default defineConfig({
    build: {
        // generate .vite/manifest.json in outDir
        manifest: true,
        rollupOptions: {
            input: 'src/main.ts',
        },
    },
})
