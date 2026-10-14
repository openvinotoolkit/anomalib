// Copyright (C) 2020-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { FlatCompat } from '@eslint/eslintrc';
import js from '@eslint/js';

import sharedEslintConfig from './eslint.shared.config.js';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);
const compat = new FlatCompat({
    baseDirectory: dirname,
    recommendedConfig: js.configs.recommended,
    allConfig: js.configs.all,
});

export default [
    {
        ignores: [...sharedEslintConfig[0].ignores, 'src/api/openapi-spec.d.ts'],
    },
    ...sharedEslintConfig,
    {
        rules: {
            'no-restricted-imports': [
                'error',
                {
                    patterns: [
                        {
                            group: ['@react-spectrum'],
                            message: 'Use component from the @geti-ui/ui folder instead.',
                        },
                        {
                            group: ['@react-types/*'],
                            message: 'Use type from the @geti-ui/ui folder instead.',
                        },
                        {
                            group: ['@spectrum-icons'],
                            message: 'Use icons from the @geti-ui/ui/icons folder instead.',
                        },
                    ],
                },
            ],
            'header/header': 'off',
        },
    },
    {
        settings: {
            'import/resolver': {
                typescript: {
                    alwaysTryTypes: true,
                    project: ['./tsconfig.json', './tests/tsconfig.json'],
                    noWarnOnMultipleProjects: true,
                },
            },
        },
    },
    ...compat.extends('plugin:playwright/playwright-test').map((config) => ({
        ...config,
        files: ['tests/features/**/*.ts', 'tests/utils/**/*.ts', 'tests/fixtures/**/*.ts', 'tests/e2e/**/*.ts'],
    })),
    {
        files: ['tests/features/**/*.ts', 'tests/utils/**/*.ts', 'tests/fixtures/**/*.ts', 'tests/e2e/**/*.ts'],

        rules: {
            'playwright/no-wait-for-selector': ['off'],
            'playwright/no-conditional-expect': ['off'],
            'playwright/no-standalone-expect': ['off'],
            'playwright/missing-playwright-await': ['warn'],
            'playwright/valid-expect': ['warn'],
            'playwright/no-useless-not': ['warn'],
            'playwright/no-page-pause': ['warn'],
            'playwright/prefer-to-have-length': ['warn'],
            'playwright/no-conditional-in-test': ['off'],
            'playwright/expect-expect': ['off'],
            'playwright/no-skipped-test': ['off'],
            'playwright/no-wait-for-timeout': ['off'],
            'playwright/no-nested-step': ['off'],
        },
    },
];
