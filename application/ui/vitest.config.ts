import react from '@vitejs/plugin-react';
import svgr from 'vite-plugin-svgr';
import tsconfigPaths from 'vite-tsconfig-paths';
import { defineConfig, type Plugin } from 'vitest/config';

// Node's ESM loader can't parse binary assets that upstream ESM bundles reference at import-time.
const stubBinaryAssets = (): Plugin => ({
    name: 'stub-binary-assets',
    enforce: 'pre',
    resolveId(id) {
        if (/\.(webp|png|jpe?g|gif|avif|ico)(\?.*)?$/.test(id)) {
            return '\0binary-asset-stub';
        }
        return null;
    },
    load(id) {
        if (id === '\0binary-asset-stub') {
            return 'export default "";';
        }
        return null;
    },
});

export default defineConfig({
    envPrefix: ['PUBLIC_'],
    plugins: [
        // TODO: Review and assess whether relative paths are necessary
        tsconfigPaths(),
        stubBinaryAssets(),
        react(),
        svgr({
            svgrOptions: {
                svgo: false,
                exportType: 'named',
            },
            include: '**/*.svg',
        }),
    ],
    test: {
        environment: 'jsdom',
        // This is needed to use globals like describe or expect
        globals: true,
        include: ['./src/**/*.test.{ts,tsx}'],
        setupFiles: './src/setup-tests.ts',
        watch: false,
        server: {
            deps: {
                inline: [
                    /@react-spectrum\/.*/,
                    /@spectrum-icons\/.*/,
                    /@adobe\/react-spectrum\/.*/,
                    // `@geti-ui/ui` ships ESM that imports React Spectrum's raw `.css`
                    // files. It must be inlined so Vite's transform handles those CSS
                    // imports; otherwise Node loads it natively and throws
                    // `Unknown file extension ".css"`.
                    /@geti-ui\/.*/,
                ],
            },
        },
    },
});
