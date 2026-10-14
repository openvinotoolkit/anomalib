import { ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { getMockedPipeline } from '../../../../mocks/mock-pipeline';
import { Toolbar } from './toolbar';

describe('Toolbar', () => {
    const renderApp = ({ pipelineConfig = {} }: { pipelineConfig?: Partial<SchemaPipeline> }) => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
            )
        );

        return render(
            <ThemeProvider>
                <QueryClientProvider client={new QueryClient()}>
                    <MemoryRouter initialEntries={['/projects/123/inspect']}>
                        <Routes>
                            <Route path='/projects/:projectId/inspect' element={<Toolbar />} />
                        </Routes>
                    </MemoryRouter>
                </QueryClientProvider>
            </ThemeProvider>
        );
    };

    describe('InferenceDevices', () => {
        it.skip('renders when model is configured', async () => {
            renderApp({
                pipelineConfig: {
                    model: {
                        id: '1',
                        name: 'test-model (1)',
                        format: 'onnx',
                        project_id: '123',
                        threshold: 0.5,
                        is_ready: true,
                        train_job_id: 'train-job-1',
                        dataset_snapshot_id: '',
                        architecture: 'PatchCore',
                    },
                },
            });

            expect(await screen.findByRole('button', { name: /inference devices/i })).toBeVisible();
        });

        it('does not render when no model is configured', async () => {
            renderApp({
                pipelineConfig: {
                    model: undefined,
                },
            });

            await waitFor(() => {
                expect(screen.queryByRole('button', { name: /inference devices/i })).not.toBeInTheDocument();
            });
        });
    });
});
