// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { StreamConnectionStatus, useStreamConnection } from 'src/components/stream/stream-connection-provider';
import { server } from 'src/msw-node-setup';
import { queryClient } from 'src/query-client/query-client';

import { renderHook } from '../../../../../tests/utils';
import { STREAM_ERROR_MESSAGE, useAutoPlayStream } from './use-auto-play-stream.hook';

vi.mock('../../../../components/stream/stream-connection-provider', async () => {
    const actual = await vi.importActual('../../../../components/stream/stream-connection-provider');
    return {
        ...actual,
        useStreamConnection: vi.fn(),
    };
});

describe('useAutoPlayStream', () => {
    const renderApp = ({
        status = 'idle',
        pipelineConfig,
    }: {
        status: StreamConnectionStatus;
        pipelineConfig?: Partial<SchemaPipeline> | null;
    }) => {
        const mockedStart = vi.fn();
        vi.mocked(useStreamConnection).mockReturnValue({
            stop: vi.fn(),
            start: mockedStart,
            status,
            streamUrl: null,
            setStatus: vi.fn(),
        });

        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline({ project_id: '123', ...pipelineConfig }))
            )
        );

        renderHook(() => useAutoPlayStream(), { route: '/projects/123/inspect' });

        return mockedStart;
    };

    beforeEach(() => {
        queryClient.clear();
        vi.clearAllMocks();
    });

    it('error message', async () => {
        const mockedStart = renderApp({ status: 'failed' });

        await waitFor(() => {
            expect(screen.getByLabelText('toast')).toHaveTextContent(STREAM_ERROR_MESSAGE);
            expect(mockedStart).not.toHaveBeenCalled();
        });
    });

    it('start stream', async () => {
        const mockedStart = renderApp({ status: 'idle' });

        await waitFor(() => {
            expect(mockedStart).toHaveBeenCalledWith();
        });
    });

    it('run pipeline', async () => {
        const pipelinePatchSpy = vi.fn();
        const mockedPipeline = getMockedPipeline({});
        mockedPipeline.status = 'active';

        server.use(
            http.post('/api/projects/{project_id}/pipeline:run', () => {
                pipelinePatchSpy();
                return HttpResponse.json({}, { status: 204 });
            })
        );
        renderApp({
            status: 'connected',
            pipelineConfig: mockedPipeline,
        });

        await waitFor(() => {
            expect(pipelinePatchSpy).toHaveBeenCalled();
        });
    });

    it('does not run pipeline if pipeline status is idle', async () => {
        const pipelinePatchSpy = vi.fn();
        const mockedPipeline = getMockedPipeline({});
        mockedPipeline.status = 'idle';

        server.use(
            http.post('/api/projects/{project_id}/pipeline:run', () => {
                pipelinePatchSpy();
                return HttpResponse.json({}, { status: 204 });
            })
        );
        renderApp({
            status: 'connected',
            pipelineConfig: mockedPipeline,
        });

        await waitFor(() => {
            expect(pipelinePatchSpy).not.toHaveBeenCalled();
        });
    });

    it('does not start stream if pipeline config is null', async () => {
        const mockedStartNull = renderApp({ status: 'idle', pipelineConfig: null });
        await waitFor(() => {
            expect(mockedStartNull).not.toHaveBeenCalled();
        });
    });

    it('does not start stream if status is connecting', async () => {
        const mockedStartConnecting = renderApp({ status: 'connecting' });
        await waitFor(() => {
            expect(mockedStartConnecting).not.toHaveBeenCalled();
        });
    });

    it('does not run pipeline if pipeline status is not idle or connected', async () => {
        const pipelinePatchSpy = vi.fn();
        const mockedPipeline = getMockedPipeline({ status: 'running' });

        server.use(
            http.post('/api/projects/{project_id}/pipeline:run', () => {
                pipelinePatchSpy();
                return HttpResponse.json({}, { status: 204 });
            })
        );
        renderApp({
            status: 'connected',
            pipelineConfig: mockedPipeline,
        });
        await waitFor(() => {
            expect(pipelinePatchSpy).not.toHaveBeenCalled();
        });
    });

    it('shows error toast if pipeline run API fails', async () => {
        const mockedPipeline = getMockedPipeline({});
        mockedPipeline.status = 'active';
        server.use(
            http.post('/api/projects/{project_id}/pipeline:run', () => {
                // @ts-expect-error -- test intentionally returns error response
                return HttpResponse.json({ detail: STREAM_ERROR_MESSAGE }, { status: 500 });
            })
        );

        renderApp({
            status: 'connected',
            pipelineConfig: mockedPipeline,
        });

        await waitFor(() => {
            expect(screen.getByLabelText('toast')).toHaveTextContent(STREAM_ERROR_MESSAGE);
        });
    });
});
