// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createElement, Suspense, type ReactNode } from 'react';

import { IntelBrandedLoading, ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
    render as rtlRender,
    renderHook as rtlRenderHook,
    type RenderHookOptions,
    type RenderOptions as RTLRenderOptions,
} from '@testing-library/react';
import { NuqsAdapter } from 'nuqs/adapters/react-router/v8';
import { createMemoryRouter, RouterProvider } from 'react-router';

import { StreamConnectionProvider } from '../src/components/stream/stream-connection-provider';
import { Toast } from '../src/components/toast/toast.component';

type TestRenderOptions = RTLRenderOptions & {
    path?: string;
    queryClient?: QueryClient;
    route?: string;
};

type TestRenderHookOptions<TProps> = Omit<RenderHookOptions<TProps>, 'wrapper'> & TestRenderOptions;

const DEFAULT_PROJECT_ID = 'project-123';

const TestProviders = ({ children, queryClient }: { children?: ReactNode; queryClient: QueryClient }) => {
    return createElement(
        QueryClientProvider,
        { client: queryClient },
        createElement(
            ThemeProvider,
            null,
            createElement(
                NuqsAdapter,
                null,
                createElement(
                    StreamConnectionProvider,
                    null,
                    createElement(Suspense, { fallback: createElement(IntelBrandedLoading) }, children)
                )
            ),
            createElement(Toast)
        )
    );
};

const createTestRouter = (children: ReactNode, options: TestRenderOptions, queryClient: QueryClient) => {
    return createMemoryRouter(
        [
            {
                path: options.path ?? '/projects/:projectId/inspect',
                element: createElement(TestProviders, { queryClient }, children),
            },
        ],
        {
            initialEntries: [options.route ?? `/projects/${DEFAULT_PROJECT_ID}/inspect`],
            initialIndex: 0,
        }
    );
};

export const render = (ui: ReactNode, options: TestRenderOptions = {}) => {
    const { path, queryClient = new QueryClient(), route, ...renderOptions } = options;
    const router = createTestRouter(ui, { path, queryClient, route }, queryClient);

    return rtlRender(createElement(RouterProvider, { router }), renderOptions);
};

export const renderHook = <TProps, TResult>(
    callback: (initialProps: TProps) => TResult,
    options: TestRenderHookOptions<TProps> = {}
) => {
    const { path, queryClient = new QueryClient(), route, ...renderHookOptions } = options;

    const Wrapper = ({ children }: { children: ReactNode }) => {
        const router = createTestRouter(children, { path, queryClient, route }, queryClient);

        return createElement(RouterProvider, { router });
    };

    return rtlRenderHook(callback, { ...renderHookOptions, wrapper: Wrapper });
};
