// Copyright (C) 2020-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { QueryClient } from '@tanstack/react-query';
import { act, waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';

import { getMockedPagination } from '../../mocks/mock-pagination';
import { renderHook } from '../../tests/utils';
import { $api } from '../api/client';
import { SchemaJob } from '../api/openapi-spec';
import { http } from '../api/utils';
import { server } from '../msw-node-setup';
import { useProjectTrainingJobs } from './use-project-trainingJobs.hook';

const makeJob = (id: string, overrides: Partial<SchemaJob> = {}): SchemaJob => ({
    id,
    project_id: 'project-123',
    type: 'training',
    status: 'completed',
    progress: 0,
    payload: {},
    message: '',
    ...overrides,
});

const mockPages = (jobs: SchemaJob[], pageSize = 50) => {
    const requests: number[] = [];
    server.use(
        http.get('/api/jobs', ({ request, response }) => {
            const params = new URL(request.url).searchParams;
            const offset = Number(params.get('offset'));
            expect(params.get('limit')).toBe('50');
            requests.push(offset);
            const page = jobs.slice(offset, offset + pageSize);
            return response(200).json({
                jobs: page,
                pagination: getMockedPagination({ offset, limit: 50, count: page.length, total: jobs.length }),
            });
        })
    );
    return requests;
};

const renderJobs = (queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })) =>
    renderHook(() => useProjectTrainingJobs(), { queryClient });

describe('useProjectTrainingJobs', () => {
    afterEach(() => {
        vi.useRealTimers();
    });

    it.each([21, 51, 101])('retrieves all %i records before filtering by project', async (total) => {
        const activeJob = makeJob('active-job', { status: 'running' });
        const requests = mockPages([
            ...Array.from({ length: total - 1 }, (_, index) =>
                makeJob(`other-${index}`, { project_id: 'another-project', status: 'pending' })
            ),
            activeJob,
        ]);
        const { result } = renderJobs();

        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        expect(result.current.jobs).toEqual([activeJob]);
        expect(requests).toEqual(Array.from({ length: Math.ceil(total / 50) }, (_, index) => index * 50));
    });

    it('keeps scanning after finding an active job and preserves history', async () => {
        const jobs = [
            makeJob('active', { status: 'pending' }),
            ...Array.from({ length: 50 }, (_, index) => makeJob(`history-${index}`)),
        ];
        const requests = mockPages(jobs);
        const { result } = renderJobs();

        await waitFor(() => expect(result.current.jobs).toEqual(jobs));
        expect(requests).toEqual([0, 50]);
    });

    it.each([0, 50, 100])('stops at the boundary for %i jobs', async (total) => {
        const jobs = Array.from({ length: total }, (_, index) => makeJob(`job-${index}`));
        const requests = mockPages(jobs);
        const { result } = renderJobs();

        await waitFor(() => expect(result.current.jobs).toHaveLength(total));
        expect(requests).toEqual(Array.from({ length: Math.max(1, total / 50) }, (_, index) => index * 50));
    });

    it('advances using the actual count when the server returns short pages', async () => {
        const jobs = Array.from({ length: 5 }, (_, index) => makeJob(`job-${index}`));
        const requests = mockPages(jobs, 2);
        const { result } = renderJobs();

        await waitFor(() => expect(result.current.jobs).toEqual(jobs));
        expect(requests).toEqual([0, 2, 4]);
    });

    it('deduplicates IDs across pages without discarding jobs with missing IDs', async () => {
        const repeated = makeJob('repeated');
        const withoutId = makeJob('unused', { id: undefined });
        mockPages([repeated, withoutId, repeated, withoutId], 2);
        const { result } = renderJobs();

        await waitFor(() => expect(result.current.jobs).toEqual([repeated, withoutId, withoutId]));
    });

    it('reports non-advancing pagination instead of looping or publishing partial jobs', async () => {
        const request = vi.fn();
        server.use(
            http.get('/api/jobs', ({ response }) => {
                request();
                return response(200).json({
                    jobs: [],
                    pagination: getMockedPagination({ limit: 50, total: 1 }),
                });
            })
        );
        const { result } = renderJobs();

        await waitFor(() => expect(result.current.isError).toBe(true));
        expect(result.current.jobs).toBeUndefined();
        expect(result.current.error).toEqual(
            new Error('Unable to retrieve all jobs: pagination did not advance as expected.')
        );
        expect(request).toHaveBeenCalledTimes(1);
    });

    it('rejects a page whose offset does not match the requested offset', async () => {
        server.use(
            http.get('/api/jobs', ({ response }) =>
                response(200).json({
                    jobs: [makeJob('unexpected')],
                    pagination: getMockedPagination({ offset: 10, limit: 50, count: 1, total: 11 }),
                })
            )
        );
        const { result } = renderJobs();

        await waitFor(() => expect(result.current.isError).toBe(true));
        expect(result.current.jobs).toBeUndefined();
    });

    it('propagates a later-page API failure without publishing a partial result', async () => {
        server.use(
            http.get('/api/jobs', ({ request, response }) => {
                const offset = Number(new URL(request.url).searchParams.get('offset'));
                if (offset > 0) {
                    return response(422).json({ detail: [] });
                }
                return response(200).json({
                    jobs: [makeJob('active', { status: 'running' })],
                    pagination: getMockedPagination({ limit: 50, count: 1, total: 2 }),
                });
            })
        );
        const { result } = renderJobs();

        await waitFor(() => expect(result.current.isError).toBe(true));
        expect(result.current.error).toEqual({ detail: [] });
        expect(result.current.jobs).toBeUndefined();
    });

    it('isolates raw page queries while sharing the aggregate between project consumers', async () => {
        const first = makeJob('first');
        const second = makeJob('second', { project_id: 'another-project' });
        const requests = mockPages([first, second], 1);
        const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: Infinity } } });
        const { result } = renderHook(
            () => ({
                first: useProjectTrainingJobs(),
                second: useProjectTrainingJobs(),
                raw: $api.useQuery('get', '/api/jobs', { params: { query: { offset: 0, limit: 50 } } }),
            }),
            { queryClient }
        );

        await waitFor(() => expect(result.current.first.isSuccess && result.current.raw.isSuccess).toBe(true));
        expect(result.current.first.jobs).toEqual([first]);
        expect(result.current.second.jobs).toEqual([first]);
        expect(result.current.raw.data?.jobs).toEqual([first]);
        const otherProject = renderHook(() => useProjectTrainingJobs(), {
            queryClient,
            route: '/projects/another-project/inspect',
        });
        await waitFor(() => expect(otherProject.result.current.jobs).toEqual([second]));
        expect(requests.filter((offset) => offset === 1)).toHaveLength(1);
        expect(queryClient.getQueryCache().getAll()).toHaveLength(2);
    });

    it('refetches every page through the existing job invalidation prefix', async () => {
        const requests = mockPages([makeJob('first'), makeJob('second')], 1);
        const queryClient = new QueryClient();
        const { result } = renderJobs(queryClient);
        await waitFor(() => expect(result.current.isSuccess).toBe(true));

        await act(async () => {
            await queryClient.invalidateQueries({ queryKey: ['get', '/api/jobs'] });
        });

        expect(requests).toEqual([0, 1, 0, 1]);
    });

    it('polls a later-page active job and stops polling when it completes', async () => {
        vi.useFakeTimers();
        const requests = mockPages(
            [makeJob('history', { project_id: 'another-project' }), makeJob('active', { status: 'running' })],
            1
        );
        const { result, unmount } = renderJobs();
        await act(async () => {
            await vi.waitFor(() => expect(result.current.isSuccess).toBe(true));
        });
        expect(requests).toEqual([0, 1]);

        const completedRequests = mockPages([makeJob('history'), makeJob('active')], 1);
        await act(async () => {
            await vi.advanceTimersByTimeAsync(5_000);
        });
        await act(async () => {
            await vi.waitFor(() => expect(result.current.jobs?.at(-1)?.status).toBe('completed'));
        });
        expect(completedRequests).toEqual([0, 1]);

        await act(async () => {
            await vi.advanceTimersByTimeAsync(10_000);
        });
        expect(completedRequests).toEqual([0, 1]);
        unmount();
    });

    it('keeps idle polling at the interval requested by the footer', async () => {
        vi.useFakeTimers();
        const requests = mockPages([]);
        const { result, unmount } = renderHook(() => useProjectTrainingJobs({ refetchInterval: 10_000 }));
        await act(async () => {
            await vi.waitFor(() => expect(result.current.isSuccess).toBe(true));
        });

        await act(async () => {
            await vi.advanceTimersByTimeAsync(5_000);
        });
        expect(requests).toEqual([0]);
        await act(async () => {
            await vi.advanceTimersByTimeAsync(5_000);
        });
        await act(async () => {
            await vi.waitFor(() => expect(requests).toEqual([0, 0]));
        });
        unmount();
    });

    it('retains successful data and reports an error when a refresh fails', async () => {
        const job = makeJob('active', { status: 'running' });
        mockPages([job]);
        const { result } = renderJobs();
        await waitFor(() => expect(result.current.jobs).toEqual([job]));
        server.use(http.get('/api/jobs', () => HttpResponse.error()));

        await act(async () => {
            await result.current.refetch();
        });

        await waitFor(() => expect(result.current.isError).toBe(true));
        expect(result.current.jobs).toEqual([job]);
    });

    it('aborts in-flight requests on unmount and does not fetch subsequent pages', async () => {
        let requestedSignal: AbortSignal | undefined;
        const requestedOffsets: number[] = [];
        let releasePage!: () => void;
        const pageReady = new Promise<void>((resolve) => {
            releasePage = resolve;
        });
        server.use(
            http.get('/api/jobs', async ({ request, response }) => {
                requestedSignal = request.signal;
                requestedOffsets.push(Number(new URL(request.url).searchParams.get('offset')));
                await pageReady;
                return response(200).json({
                    jobs: [makeJob('job')],
                    pagination: getMockedPagination({ limit: 50, count: 1, total: 2 }),
                });
            })
        );
        const { unmount } = renderJobs();
        await waitFor(() => expect(requestedOffsets).toEqual([0]));

        unmount();
        releasePage();

        await waitFor(() => expect(requestedSignal?.aborted).toBe(true));
        expect(requestedOffsets).toEqual([0]);
    });

    it('cancels a superseded scan and starts the replacement from the first page', async () => {
        const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
        const initialJob = makeJob('initial');
        mockPages([initialJob]);
        const { result } = renderJobs(queryClient);
        await waitFor(() => expect(result.current.jobs).toEqual([initialJob]));

        let supersededSignal: AbortSignal | undefined;
        let releasePage!: () => void;
        const pageReady = new Promise<void>((resolve) => {
            releasePage = resolve;
        });
        server.use(
            http.get('/api/jobs', async ({ request, response }) => {
                supersededSignal = request.signal;
                await pageReady;
                return response(200).json({
                    jobs: [initialJob],
                    pagination: getMockedPagination({ limit: 50, count: 1, total: 2 }),
                });
            })
        );
        let supersededScan!: ReturnType<typeof result.current.refetch>;
        act(() => {
            supersededScan = result.current.refetch();
        });
        await waitFor(() => expect(supersededSignal).toBeDefined());
        const replacementJob = makeJob('replacement');
        const replacementOffsets = mockPages([replacementJob]);

        await act(async () => {
            await result.current.refetch();
            releasePage();
            await supersededScan;
        });

        expect(supersededSignal?.aborted).toBe(true);
        expect(replacementOffsets).toEqual([0]);
        await waitFor(() => expect(result.current.jobs).toEqual([replacementJob]));
    });
});
