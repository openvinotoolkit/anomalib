// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { renderHook, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router';
import { describe, expect, it, vi } from 'vitest';

import { getMockedPagination } from '../../../../../../mocks/mock-pagination';
import { http } from '../../../../../api/utils';
import { server } from '../../../../../msw-node-setup';
import { StatusBarProvider, useStatusBar } from '../status-bar-context';
import { TrainingStatusAdapter } from './training-status.adapter';

interface SSEIterator {
    [Symbol.asyncIterator](): AsyncGenerator<{ progress: number; message: string }>;
}

const mockFetchSSE = vi.fn<(url: string) => SSEIterator>((_url: string) => ({
    async *[Symbol.asyncIterator]() {},
}));

vi.mock('src/api/fetch-sse', () => ({
    fetchSSE: (url: string) => mockFetchSSE(url),
}));

const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={new QueryClient()}>
        <MemoryRouter initialEntries={['/projects/test-project/inspect']}>
            <Routes>
                <Route
                    path='/projects/:projectId/inspect'
                    element={
                        <StatusBarProvider>
                            <TrainingStatusAdapter />
                            {children}
                        </StatusBarProvider>
                    }
                />
            </Routes>
        </MemoryRouter>
    </QueryClientProvider>
);

describe('TrainingStatusAdapter', () => {
    it('does not set status when no training job exists', async () => {
        server.use(
            http.get('/api/jobs', ({ response }) => response(200).json({ jobs: [], pagination: getMockedPagination() }))
        );

        const { result } = renderHook(() => useStatusBar(), { wrapper });

        await waitFor(() => {
            expect(result.current.activeStatus).toBeNull();
        });
    });

    it('sets training status when training job is running', async () => {
        const trainingJob = {
            id: 'job-1',
            project_id: 'test-project',
            type: 'training' as const,
            status: 'running' as const,
            progress: 45,
            message: 'Epoch 5/10',
            payload: { model_name: 'EfficientAd' },
        };
        server.use(
            http.get('/api/jobs', ({ response }) =>
                response(200).json({ jobs: [trainingJob], pagination: getMockedPagination() })
            )
        );

        const { result } = renderHook(() => useStatusBar(), { wrapper });

        await waitFor(() => {
            expect(result.current.activeStatus).toEqual(
                expect.objectContaining({
                    id: 'training',
                    type: 'training',
                    message: 'Training EfficientAd...',
                    detail: 'Epoch 5/10',
                    progress: 45,
                    variant: 'info',
                    isCancellable: true,
                })
            );
        });
    });

    it('sets training status when training job is pending', async () => {
        const trainingJob = {
            id: 'job-2',
            project_id: 'test-project',
            type: 'training' as const,
            status: 'pending' as const,
            progress: 0,
            message: 'Waiting...',
            payload: { model_name: 'Padim' },
        };
        server.use(
            http.get('/api/jobs', ({ response }) =>
                response(200).json({ jobs: [trainingJob], pagination: getMockedPagination() })
            )
        );

        const { result } = renderHook(() => useStatusBar(), { wrapper });

        await waitFor(() => {
            expect(result.current.activeStatus?.message).toBe('Training Padim...');
        });
    });

    it('ignores training jobs from other projects', async () => {
        const trainingJob = {
            id: 'job-3',
            project_id: 'other-project',
            type: 'training' as const,
            status: 'running' as const,
            progress: 50,
            message: 'Training...',
            payload: { model_name: 'Test' },
        };
        server.use(
            http.get('/api/jobs', ({ response }) =>
                response(200).json({ jobs: [trainingJob], pagination: getMockedPagination() })
            )
        );

        const { result } = renderHook(() => useStatusBar(), { wrapper });

        await waitFor(() => {
            expect(result.current.activeStatus).toBeNull();
        });
    });

    it('ignores completed or failed training jobs', async () => {
        const completedJob = {
            id: 'job-4',
            project_id: 'test-project',
            type: 'training' as const,
            status: 'completed' as const,
            progress: 100,
            message: 'Done',
            payload: { model_name: 'Test' },
        };
        server.use(
            http.get('/api/jobs', ({ response }) =>
                response(200).json({ jobs: [completedJob], pagination: getMockedPagination() })
            )
        );

        const { result } = renderHook(() => useStatusBar(), { wrapper });

        await waitFor(() => {
            expect(result.current.activeStatus).toBeNull();
        });
    });

    it('calls cancelJob when onCancel is triggered', async () => {
        const cancelJobSpy = vi.fn();
        const trainingJob = {
            id: 'job-5',
            project_id: 'test-project',
            type: 'training' as const,
            status: 'running' as const,
            progress: 30,
            message: 'Training...',
            payload: { model_name: 'EfficientAd' },
        };
        server.use(
            http.get('/api/jobs', ({ response }) =>
                response(200).json({ jobs: [trainingJob], pagination: getMockedPagination() })
            ),
            http.post('/api/jobs/{job_id}:cancel', ({ params }) => {
                cancelJobSpy(params.job_id);
                return HttpResponse.json({}, { status: 204 });
            })
        );

        const { result } = renderHook(() => useStatusBar(), { wrapper });

        await waitFor(() => {
            expect(result.current.activeStatus?.onCancel).toBeDefined();
        });

        result.current.activeStatus?.onCancel?.();

        await waitFor(() => {
            expect(cancelJobSpy).toHaveBeenCalledWith('job-5');
        });
    });

    it('uses SSE progress data when available', async () => {
        const trainingJob = {
            id: 'job-6',
            project_id: 'test-project',
            type: 'training' as const,
            status: 'running' as const,
            progress: 10,
            message: 'From API',
            payload: { model_name: 'Patchcore' },
        };

        mockFetchSSE.mockReturnValue({
            async *[Symbol.asyncIterator]() {
                yield { progress: 75, message: 'Stage: train' };
            },
        });

        server.use(
            http.get('/api/jobs', ({ response }) =>
                response(200).json({ jobs: [trainingJob], pagination: getMockedPagination() })
            )
        );

        const { result } = renderHook(() => useStatusBar(), { wrapper });

        await waitFor(() => {
            expect(result.current.activeStatus).toEqual(
                expect.objectContaining({
                    progress: 75,
                    detail: 'Stage: train',
                })
            );
        });
    });
});
