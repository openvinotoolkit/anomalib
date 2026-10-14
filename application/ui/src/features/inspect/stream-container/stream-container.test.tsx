import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedMetrics } from 'mocks/mock-metrics';
import { MemoryRouter, Route, Routes } from 'react-router';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { ZoomProvider } from 'src/components/zoom/zoom';
import { server } from 'src/msw-node-setup';

import { getMockedPipeline } from '../../../../mocks/mock-pipeline';
import { StreamConnectionState, useStreamConnection } from '../../../components/stream/stream-connection-provider';
import { StreamContainer } from './stream-container';

vi.mock('../../../components/stream/stream-connection-provider', () => ({
    useStreamConnection: vi.fn(),
}));

describe('StreamContainer', () => {
    const renderApp = ({
        streamConfig = {},
        pipelineConfig = {},
        hasActiveProject = true,
    }: {
        streamConfig?: Partial<StreamConnectionState>;
        pipelineConfig?: Partial<SchemaPipeline>;
        hasActiveProject?: boolean;
    }) => {
        vi.mocked(useStreamConnection).mockReturnValue({
            stop: vi.fn(),
            start: vi.fn(),
            status: 'idle',
            streamUrl: null,
            setStatus: vi.fn(),
            ...streamConfig,
        });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
            ),
            http.get('/api/projects/{project_id}/pipeline/metrics', ({ response }) =>
                response(200).json(getMockedMetrics({}))
            )
        );

        render(
            <QueryClientProvider client={new QueryClient()}>
                <ZoomProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect/stream']}>
                        <Routes>
                            <Route
                                path='/projects/:projectId/inspect/stream'
                                element={<StreamContainer hasActiveProject={hasActiveProject} />}
                            />
                        </Routes>
                    </MemoryRouter>
                </ZoomProvider>
            </QueryClientProvider>
        );
    };

    describe('Start stream button', () => {
        it('starts the stream when clicked', async () => {
            const mockedStart = vi.fn();

            renderApp({
                streamConfig: { status: 'idle', start: mockedStart },
                pipelineConfig: { status: 'idle' },
            });

            const button = await screen.findByRole('button', { name: /Start stream/i });
            await userEvent.click(button);

            expect(mockedStart).toHaveBeenCalled();
        });

        it('does not start the stream when there is no active project', async () => {
            const mockedStart = vi.fn();

            renderApp({
                streamConfig: { status: 'idle', start: mockedStart },
                pipelineConfig: { status: 'idle' },
                hasActiveProject: false,
            });

            const button = await screen.findByRole('button', { name: /Start stream/i });
            await userEvent.click(button);

            expect(mockedStart).not.toHaveBeenCalled();
        });
    });

    it('renders stream while connecting', async () => {
        renderApp({ streamConfig: { status: 'connecting', streamUrl: '/api/stream' } });

        expect(await screen.findByLabelText('stream player')).toBeVisible();
    });

    it('renders stream when connected', async () => {
        renderApp({ streamConfig: { status: 'connected', streamUrl: '/api/stream' } });

        expect(await screen.findByLabelText('stream player')).toBeVisible();
    });
});
