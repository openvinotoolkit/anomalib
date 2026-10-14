import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { getMockedPipeline } from '../../../../../mocks/mock-pipeline';
import { AnomalyMap } from './anomaly-map.component';

describe('AnomalyMap', () => {
    const renderApp = ({ pipelineConfig = {} }: { pipelineConfig?: Partial<SchemaPipeline> } = {}) => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', () => {
                if (pipelineConfig === null) {
                    return HttpResponse.json(null);
                }
                return HttpResponse.json(getMockedPipeline(pipelineConfig));
            })
        );

        return render(
            <QueryClientProvider client={new QueryClient()}>
                <MemoryRouter initialEntries={['/projects/123/inspect']}>
                    <Routes>
                        <Route path='/projects/:projectId/inspect' element={<AnomalyMap />} />
                    </Routes>
                </MemoryRouter>
            </QueryClientProvider>
        );
    };

    it('disables button if no pipeline', async () => {
        renderApp({ pipelineConfig: { status: 'idle' } });

        expect(await screen.findByRole('switch')).toBeDisabled();
    });

    it('renders enabled switch if pipeline is running', async () => {
        renderApp({ pipelineConfig: { status: 'running' } });

        expect(await screen.findByRole('switch')).toBeEnabled();
    });

    it('calls API and toggles switch', async () => {
        const requestSpy = vi.fn();

        server.use(
            http.patch('/api/projects/{project_id}/pipeline', async ({ response }) => {
                requestSpy();
                return response(200).json(getMockedPipeline());
            })
        );

        renderApp({ pipelineConfig: { status: 'running' } });

        await userEvent.click(await screen.findByRole('switch'));

        await waitFor(() => {
            expect(requestSpy).toHaveBeenCalled();
        });
    });
});
