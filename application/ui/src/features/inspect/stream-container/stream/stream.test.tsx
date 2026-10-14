// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Toast } from '@anomalib-studio/toast';
import { ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedMediaItem } from 'mocks/mock-media-item';
import { getMockedMetrics } from 'mocks/mock-metrics';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router';
import { SchemaPipeline, SchemaPipelineMetrics } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { ZoomProvider } from 'src/components/zoom/zoom';
import { server } from 'src/msw-node-setup';

import { StreamConnectionState, useStreamConnection } from '../../../../components/stream/stream-connection-provider';
import { Stream } from './stream.component';

vi.mock('../../../../components/stream/stream-connection-provider', async () => {
    const actual = await vi.importActual('../../../../components/stream/stream-connection-provider');
    return { ...actual, useStreamConnection: vi.fn() };
});

interface RenderStreamOptions {
    streamOverrides?: Partial<NonNullable<StreamConnectionState>>;
    pipelineConfig?: Partial<SchemaPipeline>;
    metricsConfig?: Partial<SchemaPipelineMetrics>;
}

const renderStream = ({ streamOverrides = {}, pipelineConfig = {}, metricsConfig = {} }: RenderStreamOptions = {}) => {
    vi.mocked(useStreamConnection).mockReturnValue({
        status: 'connected',
        streamUrl: '/api/stream',
        start: vi.fn(),
        stop: vi.fn(),
        setStatus: vi.fn(),
        ...streamOverrides,
    });

    server.use(
        http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
            response(200).json(getMockedPipeline(pipelineConfig))
        ),
        http.get('/api/projects/{project_id}/pipeline/metrics', ({ response }) =>
            response(200).json(getMockedMetrics(metricsConfig))
        )
    );

    return render(
        <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
            <ThemeProvider>
                <ZoomProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect/stream']}>
                        <Routes>
                            <Route path='/projects/:projectId/inspect/stream' element={<Stream />} />
                        </Routes>
                    </MemoryRouter>
                </ZoomProvider>
                <Toast />
            </ThemeProvider>
        </QueryClientProvider>
    );
};

describe('Stream', () => {
    describe('rendering by status', () => {
        it('renders the stream image with the provided streamUrl', async () => {
            const streamUrl = '/api/stream?ts=1';
            renderStream({ streamOverrides: { status: 'connecting', streamUrl } });

            const image = (await screen.findByLabelText('stream player')) as HTMLImageElement;
            expect(image).toBeVisible();
            expect(image.getAttribute('src')).toBe(streamUrl);
        });

        it('does not render the Capture button or Fps when status is "connecting"', async () => {
            renderStream({ streamOverrides: { status: 'connecting' } });

            await screen.findByLabelText('stream player');
            expect(screen.queryByRole('button', { name: /Capture/i })).not.toBeInTheDocument();
            expect(screen.queryByText(/fps/i)).not.toBeInTheDocument();
        });

        it('renders the Capture button and Fps when status is "connected"', async () => {
            renderStream({ streamOverrides: { status: 'connected' } });

            expect(await screen.findByRole('button', { name: /Capture/i })).toBeVisible();
            expect(await screen.findByText(/fps/i)).toBeVisible();
        });
    });

    describe('handleStreamLoad', () => {
        it('does not call setStatus when streamUrl is null (guard against spurious loads)', async () => {
            const setStatus = vi.fn();
            renderStream({ streamOverrides: { status: 'connecting', streamUrl: null, setStatus } });

            const image = await screen.findByLabelText('stream player');
            fireEvent.load(image);

            expect(setStatus).not.toHaveBeenCalled();
        });
    });

    describe('handleStreamError', () => {
        it('does not call setStatus or show a toast when streamUrl is null', async () => {
            const setStatus = vi.fn();
            renderStream({ streamOverrides: { status: 'connecting', streamUrl: null, setStatus } });

            const image = await screen.findByLabelText('stream player');
            fireEvent.error(image);

            expect(setStatus).not.toHaveBeenCalled();
            // Flush pending microtasks so a stray toast would have surfaced.
            await Promise.resolve();
            expect(screen.queryByText('Stream connection failed')).not.toBeInTheDocument();
        });
    });

    describe('pause button', () => {
        it('clears the image src to "data:," and calls stop', async () => {
            const stop = vi.fn().mockResolvedValue(undefined);
            renderStream({ streamOverrides: { status: 'connected', streamUrl: '/api/stream', stop } });

            const image = (await screen.findByLabelText('stream player')) as HTMLImageElement;
            await userEvent.click(screen.getByRole('button', { name: /Pause stream/i }));

            expect(image.getAttribute('src')).toBe('data:,');
            expect(stop).toHaveBeenCalledTimes(1);
        });
    });

    describe('capture button', () => {
        it('calls the capture endpoint on click', async () => {
            const captureSpy = vi.fn();
            renderStream({ streamOverrides: { status: 'connected', streamUrl: '/api/stream' } });
            server.use(
                http.get('/api/projects/{project_id}/capture', () => {
                    captureSpy();
                    return HttpResponse.json(getMockedMediaItem({}), { status: 201 });
                })
            );

            await userEvent.click(await screen.findByRole('button', { name: /Capture/i }));

            await waitFor(() => {
                expect(captureSpy).toHaveBeenCalledTimes(1);
            });
        });

        it('shows an error toast when the capture endpoint fails', async () => {
            // handleCaptureFrame awaits mutateAsync, which re-throws on 500 and surfaces as an unhandled rejection.
            const swallow = (event: PromiseRejectionEvent | { preventDefault?: () => void }) =>
                event.preventDefault?.();
            window.addEventListener('unhandledrejection', swallow);
            process.on('unhandledRejection', swallow);

            try {
                renderStream({ streamOverrides: { status: 'connected', streamUrl: '/api/stream' } });
                server.use(
                    http.get('/api/projects/{project_id}/capture', () =>
                        // @ts-expect-error -- test intentionally returns error response
                        HttpResponse.json({ detail: 'boom' }, { status: 500 })
                    )
                );

                await userEvent.click(await screen.findByRole('button', { name: /Capture/i }));

                expect(await screen.findByText('Failed to upload 1 item')).toBeVisible();
            } finally {
                window.removeEventListener('unhandledrejection', swallow);
                process.off('unhandledRejection', swallow);
            }
        });
    });
});
