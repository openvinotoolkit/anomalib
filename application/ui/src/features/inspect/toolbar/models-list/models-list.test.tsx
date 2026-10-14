// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router';
import { SchemaModel, SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { getMockedModel } from '../../../../../mocks/mock-model';
import { getMockedPipeline } from '../../../../../mocks/mock-pipeline';
import { ModelsList } from './models-list.component';

describe('ModelsList', () => {
    const renderApp = ({
        models = [],
        pipelineConfig = {},
    }: {
        models?: SchemaModel[];
        pipelineConfig?: Partial<SchemaPipeline>;
    } = {}) => {
        server.use(
            http.get('/api/projects/{project_id}/models', ({ response }) =>
                response(200).json({
                    models,
                    pagination: {
                        offset: 0,
                        limit: 0,
                        count: 0,
                        total: 0,
                    },
                })
            ),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
            )
        );

        return render(
            <QueryClientProvider client={new QueryClient()}>
                <ThemeProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect']}>
                        <Routes>
                            <Route path='/projects/:projectId/inspect' element={<ModelsList />} />
                        </Routes>
                    </MemoryRouter>
                </ThemeProvider>
            </QueryClientProvider>
        );
    };

    it('renders empty state when no models are available', async () => {
        renderApp({ models: [] });

        expect(await screen.findByText('No models trained yet')).toBeVisible();
    });

    it('renders list of models', async () => {
        renderApp({
            models: [
                getMockedModel({ id: '1', name: 'Model 1' }),
                getMockedModel({ id: '2', name: 'Model 2' }),
                getMockedModel({ id: '3', name: 'Model 3' }),
            ],
        });

        expect(await screen.findByText('Model 1')).toBeVisible();
        expect(await screen.findByText('Model 2')).toBeVisible();
        expect(await screen.findByText('Model 3')).toBeVisible();
    });

    it('patches pipeline when selecting a model', async () => {
        const requestSpy = vi.fn();

        server.use(
            http.patch('/api/projects/{project_id}/pipeline', async ({ response }) => {
                requestSpy();
                return response(200).json(getMockedPipeline());
            })
        );

        renderApp({
            models: [getMockedModel({ id: '1', name: 'Model 1' }), getMockedModel({ id: '2', name: 'Model 2' })],
            pipelineConfig: { model: getMockedModel({ id: '1', name: 'Model 1' }) },
        });

        await userEvent.click(await screen.findByText('Model 2'));

        await waitFor(() => {
            expect(requestSpy).toHaveBeenCalled();
        });
    });
});
