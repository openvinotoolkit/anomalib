// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { IntelBrandedLoading } from '@geti-ui/ui';
import { createBrowserRouter, Navigate, Outlet } from 'react-router';

import { Toast } from '../components/toast/toast.component';
import { $api } from './../api/client';
import { ErrorPage } from './../components/error-page/error-page';
import { Inspect } from './inspect/inspect';
import { Layout } from './layout';
import { OpenApi } from './openapi/openapi';
import { paths } from './paths';
import { Welcome } from './welcome';

const Redirect = () => {
    const { data } = $api.useSuspenseQuery('get', '/api/projects');

    if (data.projects.length === 0) {
        return <Navigate to={paths.welcome({})} replace />;
    }

    const projectId = data.projects.at(0)?.id ?? '1';
    return <Navigate to={`${paths.project({ projectId })}?mode=Dataset`} replace />;
};

export const router = createBrowserRouter([
    {
        path: paths.root.pattern,
        errorElement: <ErrorPage />,
        element: (
            <Suspense fallback={<IntelBrandedLoading />}>
                <div data-react-aria-top-layer='true'>
                    <Toast />
                </div>

                <Outlet />
            </Suspense>
        ),
        children: [
            {
                index: true,
                element: <Redirect />,
            },
            {
                path: paths.welcome.pattern,
                element: <Welcome />,
            },
            {
                path: paths.project.pattern,
                element: <Layout />,
                children: [
                    {
                        index: true,
                        element: <Inspect />,
                    },
                ],
            },
            {
                path: paths.openapi.pattern,
                element: <OpenApi />,
            },
            {
                path: '*',
                element: <Redirect />,
            },
        ],
    },
]);
