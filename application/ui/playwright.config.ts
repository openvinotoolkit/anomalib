import { defineConfig, devices } from '@playwright/test';

const CI = !!process.env.CI;

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
    testDir: './tests',
    /* Run tests in files in parallel */
    fullyParallel: true,
    /* Fail the build on CI if you accidentally left test.only in the source code. */
    forbidOnly: CI,
    /* Retry on CI only */
    retries: process.env.CI ? 2 : 0,
    /* Opt out of parallel tests on CI. */
    workers: process.env.CI ? 1 : undefined,
    /* Reporter to use. See https://playwright.dev/docs/test-reporters */
    reporter: [[CI ? 'github' : 'list'], ['html', { open: 'never' }]],
    use: {
        baseURL: 'http://localhost:3000',
        trace: CI ? 'on-first-retry' : 'on',
        video: CI ? 'on-first-retry' : 'on',
        launchOptions: {
            slowMo: 100,
            headless: true,
        },
        timezoneId: 'UTC',
        actionTimeout: CI ? 10000 : 5000,
        navigationTimeout: CI ? 10000 : 5000,
    },

    /* Configure projects for major browsers */
    projects: [
        {
            name: 'Component tests',
            use: { ...devices['Desktop Chrome'] },
        },
    ],

    /* Run your local dev server before starting the tests */
    webServer: {
        command: CI ? 'npx serve -s dist -p 3000' : 'npm run start',
        name: 'client',
        url: 'http://localhost:3000',
        reuseExistingServer: CI === false,
    },
});
