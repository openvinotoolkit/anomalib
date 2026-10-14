// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { toast } from 'src/components/toast/toast.component';
import { server } from 'src/msw-node-setup';

import { getMockedPipeline } from '../../../../../mocks/mock-pipeline';
import { TogglePipelineButton } from './toggle-pipeline-button.component';

vi.mock('src/components/toast/toast.component', () => ({
    toast: vi.fn(),
}));

describe('TogglePipelineButton', () => {
    const renderApp = ({ pipelineConfig = {} }: { pipelineConfig?: Partial<SchemaPipeline> } = {}) => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
            )
        );

        return render(
            <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
                <ThemeProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect']}>
                        <Routes>
                            <Route path='/projects/:projectId/inspect' element={<TogglePipelineButton />} />
                        </Routes>
                    </MemoryRouter>
                </ThemeProvider>
            </QueryClientProvider>
        );
    };

    beforeEach(() => {
        vi.mocked(toast).mockClear();
    });

    it('renders selected switch with "enabled" label when pipeline is running', async () => {
        renderApp({ pipelineConfig: { status: 'running' } });

        const toggle = await screen.findByRole('switch');
        expect(toggle).toBeChecked();
        expect(screen.getByText(/pipeline enabled/i)).toBeVisible();
    });

    it('renders unselected switch with "disabled" label when pipeline is not running', async () => {
        renderApp({ pipelineConfig: { status: 'idle' } });

        const toggle = await screen.findByRole('switch');
        expect(toggle).not.toBeChecked();
        expect(screen.getByText(/pipeline disabled/i)).toBeVisible();
    });

    it('calls the activate endpoint and shows a success toast when enabling', async () => {
        const activateSpy = vi.fn();
        const runSpy = vi.fn();

        server.use(
            http.post('/api/projects/{project_id}/pipeline:activate', () => {
                activateSpy();
                return HttpResponse.json({}, { status: 204 });
            }),
            http.post('/api/projects/{project_id}/pipeline:run', () => {
                runSpy();
                return HttpResponse.json({}, { status: 204 });
            })
        );

        renderApp({ pipelineConfig: { status: 'idle' } });

        await userEvent.click(await screen.findByRole('switch'));

        await waitFor(() => {
            expect(runSpy).toHaveBeenCalled();
            expect(activateSpy).toHaveBeenCalled();
        });

        await waitFor(() => {
            expect(toast).toHaveBeenCalledWith({
                type: 'success',
                message: 'Pipeline enabled successfully',
            });
        });
    });

    it('calls the disable endpoint and shows a success toast when disabling', async () => {
        const disableSpy = vi.fn();

        server.use(
            http.post('/api/projects/{project_id}/pipeline:disable', () => {
                disableSpy();
                return HttpResponse.json({}, { status: 204 });
            })
        );

        renderApp({ pipelineConfig: { status: 'running' } });

        await userEvent.click(await screen.findByRole('switch'));

        await waitFor(() => {
            expect(disableSpy).toHaveBeenCalledTimes(1);
        });

        await waitFor(() => {
            expect(toast).toHaveBeenCalledWith({
                type: 'success',
                message: 'Pipeline disabled successfully',
            });
        });
    });

    it('disables the switch while a mutation is pending', async () => {
        server.use(
            http.post('/api/projects/{project_id}/pipeline:activate', async () => {
                await new Promise((resolve) => setTimeout(resolve, 100));
                return HttpResponse.json({}, { status: 204 });
            })
        );

        renderApp({ pipelineConfig: { status: 'idle' } });

        const toggle = await screen.findByRole('switch');
        await userEvent.click(toggle);

        await waitFor(() => {
            expect(screen.getByRole('switch')).toBeDisabled();
        });
    });
});
