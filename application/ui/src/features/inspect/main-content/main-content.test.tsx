import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { ZoomProvider } from 'src/components/zoom/zoom';
import { server } from 'src/msw-node-setup';

import { getMockedPipeline } from '../../../../mocks/mock-pipeline';
import { StreamConnectionState, useStreamConnection } from '../../../components/stream/stream-connection-provider';
import { MainContent } from './main-content.component';
import { SOURCE_MESSAGE } from './source-sink-message/source-sink-message.component';

vi.mock('../../../components/stream/stream-connection-provider', () => ({
    useStreamConnection: vi.fn(),
}));

describe('MainContent', () => {
    const renderApp = ({
        streamConfig = {},
        pipelineConfig = {},
        activePipelineConfig = {},
    }: {
        streamConfig?: Partial<StreamConnectionState>;
        pipelineConfig?: Partial<SchemaPipeline>;
        activePipelineConfig?: Partial<SchemaPipeline> | null;
    }) => {
        vi.mocked(useStreamConnection).mockReturnValue({
            start: vi.fn(),
            status: 'idle',
            stop: vi.fn(),
            streamUrl: null,
            setStatus: vi.fn(),
            ...streamConfig,
        });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
            ),
            http.get('/api/active-pipeline', ({ response }) => {
                if (activePipelineConfig === null) {
                    return response(200).json(null);
                }
                return response(200).json(getMockedPipeline({ project_id: '123', ...activePipelineConfig }));
            })
        );

        return render(
            <QueryClientProvider client={new QueryClient()}>
                <ZoomProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect']}>
                        <Routes>
                            <Route path='/projects/:projectId/inspect' element={<MainContent />} />
                        </Routes>
                    </MemoryRouter>
                </ZoomProvider>
            </QueryClientProvider>
        );
    };

    describe('SinkMessage', () => {
        it('renders when no source is configured', async () => {
            renderApp({ pipelineConfig: { source: undefined } });

            expect(await screen.findByText(SOURCE_MESSAGE)).toBeVisible();
        });

        it('does not render SinkMessage when no sink is configured', async () => {
            renderApp({ pipelineConfig: { sink: undefined } });

            await waitFor(() => {
                expect(screen.queryByText(SOURCE_MESSAGE)).not.toBeInTheDocument();
            });
        });

        it('renders when both source and sink are missing', async () => {
            renderApp({ pipelineConfig: { source: undefined, sink: undefined } });

            expect(await screen.findByText(SOURCE_MESSAGE)).toBeVisible();
        });
    });

    describe('EnableProject', () => {
        it('renders when another project is active', async () => {
            renderApp({
                pipelineConfig: { project_id: '123' },
                activePipelineConfig: { project_id: '456' },
            });

            expect(await screen.findByRole('button', { name: /Activate project/i })).toBeVisible();
        });

        it('does not render EnableProject when current project is active', async () => {
            renderApp({
                pipelineConfig: { project_id: '123' },
                activePipelineConfig: { project_id: '123' },
            });

            await waitFor(() => {
                expect(screen.queryByRole('button', { name: /Activate project/i })).not.toBeInTheDocument();
            });
        });

        it('does not render EnableProject when no active pipeline', async () => {
            renderApp({
                pipelineConfig: { project_id: '123' },
                activePipelineConfig: null,
            });

            await waitFor(() => {
                expect(screen.queryByRole('button', { name: /Activate project/i })).not.toBeInTheDocument();
            });
        });
    });

    describe('StreamContainer', () => {
        it('renders when source is configured', async () => {
            renderApp({
                pipelineConfig: { status: 'idle' },
                streamConfig: { status: 'idle' },
            });

            expect(await screen.findByRole('button', { name: /Start stream/i })).toBeVisible();
        });

        it('renders when current project is active', async () => {
            renderApp({
                streamConfig: { status: 'idle' },
                pipelineConfig: { project_id: '123', status: 'running' },
                activePipelineConfig: { project_id: '123' },
            });

            expect(await screen.findByRole('button', { name: /Start stream/i })).toBeVisible();
        });
    });
});
