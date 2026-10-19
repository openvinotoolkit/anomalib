// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DialogContainer } from '@geti-ui/ui';
import { QueryClient } from '@tanstack/react-query';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPagination } from 'mocks/mock-pagination';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { TRAINABLE_MODELS } from 'mocks/mock-trainable-models';
import { delay, HttpResponse } from 'msw';
import { SchemaJob, SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { queryClient as applicationQueryClient } from 'src/query-client/query-client';

import { render } from '../../../../tests/utils';
import { TrainModelDialog } from './train-model-dialog.component';

type DeviceType = 'cpu' | 'xpu' | 'cuda';

interface MockDeviceInfo {
    type: DeviceType;
    name: string;
    memory: number | null;
    index: number | null;
}

const DEFAULT_MOCK_DEVICES: MockDeviceInfo[] = [
    { type: 'cpu', name: 'CPU', memory: null, index: null },
    { type: 'cuda', name: 'NVIDIA RTX 4090', memory: 8589934592, index: 0 },
];

const MULTI_GPU_DEVICES: MockDeviceInfo[] = [
    { type: 'cpu', name: 'CPU', memory: null, index: null },
    { type: 'cuda', name: 'NVIDIA RTX 4090', memory: 8589934592, index: 0 },
    { type: 'cuda', name: 'NVIDIA RTX 3090', memory: 8589934592, index: 1 },
];

const MULTI_XPU_DEVICES: MockDeviceInfo[] = [
    { type: 'cpu', name: 'CPU', memory: null, index: null },
    { type: 'xpu', name: 'Intel(R) Graphics', memory: 30673268736, index: 0 },
    { type: 'xpu', name: 'Intel(R) Graphics', memory: 30673268736, index: 1 },
];

interface RenderDialogOptions {
    devices?: MockDeviceInfo[];
    jobs?: SchemaJob[];
    pipelineConfig?: Partial<SchemaPipeline>;
    jobsHandler?: ReturnType<typeof http.get>;
    queryClient?: QueryClient;
}

describe('TrainModelDialog', () => {
    const closeMock = vi.fn();

    const renderDialog = ({
        devices = DEFAULT_MOCK_DEVICES,
        jobs = [],
        pipelineConfig,
        jobsHandler,
        queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } }),
    }: RenderDialogOptions = {}) => {
        server.use(
            http.get('/api/trainable-models', ({ response }) =>
                response(200).json({ trainable_models: TRAINABLE_MODELS })
            ),
            http.get('/api/system/devices/training', ({ response }) => response(200).json(devices)),
            http.post('/api/jobs:train', ({ response }) => response(200).json({ job_id: 'job-123' })),
            jobsHandler ??
                http.get('/api/jobs', ({ request, response }) => {
                    const params = new URL(request.url).searchParams;
                    const offset = Number(params.get('offset'));
                    const limit = Number(params.get('limit'));
                    const page = jobs.slice(offset, offset + limit);
                    return response(200).json({
                        jobs: page,
                        pagination: getMockedPagination({ offset, limit, count: page.length, total: jobs.length }),
                    });
                }),
            http.get('/api/active-pipeline', ({ response }) =>
                response(200).json(getMockedPipeline({ status: 'idle', ...pipelineConfig }))
            )
        );

        return render(
            <DialogContainer onDismiss={closeMock}>
                <TrainModelDialog close={closeMock} />
            </DialogContainer>,
            { route: '/projects/123/inspect', queryClient }
        );
    };

    beforeEach(() => {
        closeMock.mockClear();
    });

    describe('Form Validation', () => {
        it('disables Start button when no model is selected', async () => {
            renderDialog();

            expect(await screen.findByText('Train model')).toBeVisible();

            const startButton = screen.getByRole('button', { name: /start/i });
            expect(startButton).toBeDisabled();
        });

        it('disables Start button when no device is selected', async () => {
            renderDialog({ devices: [] });

            expect(await screen.findByText('Train model')).toBeVisible();

            const startButton = screen.getByRole('button', { name: /start/i });
            expect(startButton).toBeDisabled();
        });

        it('enables Start button when model and device are selected', async () => {
            renderDialog();

            expect(await screen.findByText('PatchCore')).toBeVisible();

            await userEvent.click(screen.getByText('PatchCore'));

            await waitFor(() => {
                const startButton = screen.getByRole('button', { name: /start/i });
                expect(startButton).not.toBeDisabled();
            });
        });
    });

    describe('Form Submission', () => {
        it('closes dialog after successful submission', async () => {
            renderDialog();

            expect(await screen.findByText('PatchCore')).toBeVisible();
            await userEvent.click(screen.getByText('PatchCore'));

            await waitFor(() => {
                expect(screen.getByRole('button', { name: /start/i })).not.toBeDisabled();
            });

            await userEvent.click(screen.getByRole('button', { name: /start/i }));

            await waitFor(() => {
                expect(closeMock).toHaveBeenCalled();
            });
        });

        it('refreshes the complete job query after submitting training', async () => {
            applicationQueryClient.clear();
            const offsets: number[] = [];
            let submitted = false;
            const dialog = renderDialog({
                queryClient: applicationQueryClient,
                jobsHandler: http.get('/api/jobs', ({ request, response }) => {
                    const offset = Number(new URL(request.url).searchParams.get('offset'));
                    offsets.push(offset);
                    return response(200).json({
                        jobs: [
                            {
                                id: `job-${offset}`,
                                project_id: offset === 0 ? 'another-project' : '123',
                                type: 'training',
                                status: submitted ? 'pending' : 'completed',
                                payload: {},
                                progress: 0,
                                message: '',
                            },
                        ],
                        pagination: getMockedPagination({ offset, limit: 50, count: 1, total: 2 }),
                    });
                }),
            });
            server.use(
                http.post('/api/jobs:train', ({ response }) => {
                    submitted = true;
                    return response(200).json({ job_id: 'job-1' });
                })
            );

            await userEvent.click(await screen.findByText('PatchCore'));
            await waitFor(() => expect(offsets).toEqual([0, 1]));
            await userEvent.click(screen.getByRole('button', { name: /^start$/i }));

            await waitFor(() => expect(offsets).toEqual([0, 1, 0, 1]));
            expect(await screen.findByText('Training already queued')).toBeVisible();
            expect(closeMock).toHaveBeenCalled();
            dialog.unmount();
            applicationQueryClient.clear();
        });
    });

    describe('Multi-device support', () => {
        it('shows distinct entries for multiple CUDA devices', async () => {
            renderDialog({ devices: MULTI_GPU_DEVICES });

            expect(await screen.findByText('Train model')).toBeVisible();

            // Both GPUs should appear with their name and index
            // Picker renders both <option> and visible text, so use getAllByText
            expect(screen.getAllByText('NVIDIA RTX 4090 [0]').length).toBeGreaterThanOrEqual(1);
            expect(screen.getAllByText('NVIDIA RTX 3090 [1]').length).toBeGreaterThanOrEqual(1);
        });

        it('shows distinct entries for multiple XPU devices with same name', async () => {
            renderDialog({ devices: MULTI_XPU_DEVICES });

            expect(await screen.findByText('Train model')).toBeVisible();

            // Both XPUs should appear with index to distinguish them
            expect(screen.getAllByText('Intel(R) Graphics [0]').length).toBeGreaterThanOrEqual(1);
            expect(screen.getAllByText('Intel(R) Graphics [1]').length).toBeGreaterThanOrEqual(1);
        });

        it('shows CPU without index suffix', async () => {
            renderDialog();

            expect(await screen.findByText('Train model')).toBeVisible();

            // CPU has index=null, so no [index] suffix
            // NVIDIA RTX 4090 should show with index
            expect(screen.getAllByText('CPU').length).toBeGreaterThanOrEqual(1);
            expect(screen.getAllByText('NVIDIA RTX 4090 [0]').length).toBeGreaterThanOrEqual(1);
        });
    });

    describe('Warnings', () => {
        const makeJob = (status: SchemaJob['status']): SchemaJob => ({
            id: 'job-1',
            project_id: '123',
            type: 'training',
            status,
            progress: 0,
            payload: {},
            message: 'Job created',
        });

        const CPU_ONLY = [{ type: 'cpu' as const, name: 'CPU', memory: null, index: null }];

        it.each([['running'], ['pending']] as const)(
            'shows the queued job warning when a training job is %s',
            async (status) => {
                renderDialog({ jobs: [makeJob(status)] });

                expect(await screen.findByText('Training already queued')).toBeVisible();
                expect(screen.getByText(/same training configuration cannot be submitted again/)).toBeVisible();
            }
        );

        it('does not show the queued job warning when no active jobs exist', async () => {
            renderDialog();

            expect(await screen.findByText('Train model')).toBeVisible();
            expect(screen.queryByText('Training already queued')).not.toBeInTheDocument();
        });

        it.each(['running', 'pending'] as const)('finds a %s training job on the third page', async (status) => {
            renderDialog({
                jobs: [
                    ...Array.from({ length: 100 }, (_, index) => ({
                        ...makeJob('running'),
                        id: `other-${index}`,
                        project_id: 'another-project',
                    })),
                    makeJob(status),
                ],
            });

            expect(await screen.findByText('Training already queued')).toBeVisible();
            await waitFor(() => {
                expect(screen.queryByText(/Checking for pending/)).not.toBeInTheDocument();
            });
        });

        it.each(['completed', 'failed', 'canceled'] as const)(
            'does not warn for a %s training job after checking all pages',
            async (status) => {
                renderDialog({ jobs: [makeJob(status)] });

                await waitFor(() => {
                    expect(screen.queryByText(/Checking for pending/)).not.toBeInTheDocument();
                });
                expect(screen.queryByText('Training already queued')).not.toBeInTheDocument();
            }
        );

        it('shows checking feedback while a later page loads without blocking Start', async () => {
            let releasePage!: () => void;
            const pageReady = new Promise<void>((resolve) => {
                releasePage = resolve;
            });
            renderDialog({
                jobsHandler: http.get('/api/jobs', async ({ request, response }) => {
                    const offset = Number(new URL(request.url).searchParams.get('offset'));
                    if (offset === 1) {
                        await pageReady;
                    }
                    return response(200).json({
                        jobs: [
                            {
                                ...makeJob('running'),
                                id: `job-${offset}`,
                                project_id: offset === 0 ? 'other-project' : '123',
                            },
                        ],
                        pagination: getMockedPagination({ offset, limit: 50, count: 1, total: 2 }),
                    });
                }),
            });

            expect(await screen.findByText(/Checking for pending/)).toBeVisible();
            await userEvent.click(await screen.findByText('PatchCore'));
            expect(screen.getByRole('button', { name: /^start$/i })).toBeEnabled();
            expect(screen.queryByText('Training already queued')).not.toBeInTheDocument();

            releasePage();
            expect(await screen.findByText('Training already queued')).toBeVisible();
        });

        it('exposes a later-page error and retries without blocking Start', async () => {
            let shouldFail = true;
            renderDialog({
                jobsHandler: http.get('/api/jobs', ({ request, response }) => {
                    const offset = Number(new URL(request.url).searchParams.get('offset'));
                    if (offset === 1 && shouldFail) {
                        return HttpResponse.error();
                    }
                    return response(200).json({
                        jobs: [
                            {
                                ...makeJob('pending'),
                                id: `job-${offset}`,
                                project_id: offset === 0 ? 'other-project' : '123',
                            },
                        ],
                        pagination: getMockedPagination({ offset, limit: 50, count: 1, total: 2 }),
                    });
                }),
            });

            expect(await screen.findByText('Unable to check training jobs')).toBeVisible();
            expect(screen.queryByText('Training already queued')).not.toBeInTheDocument();
            await userEvent.click(await screen.findByText('PatchCore'));
            expect(screen.getByRole('button', { name: /^start$/i })).toBeEnabled();

            shouldFail = false;
            await userEvent.click(screen.getByRole('button', { name: 'Retry job check' }));
            expect(await screen.findByText('Training already queued')).toBeVisible();
            expect(screen.queryByText('Unable to check training jobs')).not.toBeInTheDocument();
        });

        it('checks again on opening even when cached jobs are fresh', async () => {
            const queryClient = new QueryClient({ defaultOptions: { queries: { staleTime: 300_000 } } });
            const firstDialog = renderDialog({ queryClient });
            await waitFor(() => expect(queryClient.isFetching()).toBe(0));
            firstDialog.unmount();

            renderDialog({ queryClient, jobs: [makeJob('pending')] });

            expect(await screen.findByText('Training already queued')).toBeVisible();
        });

        it('retains a known warning when refreshing the job check fails', async () => {
            const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
            renderDialog({ queryClient, jobs: [makeJob('running')] });
            expect(await screen.findByText('Training already queued')).toBeVisible();
            server.use(
                http.get('/api/jobs', async () => {
                    await delay(10);
                    return HttpResponse.error();
                })
            );

            void queryClient.invalidateQueries({ queryKey: ['get', '/api/jobs'] });

            expect(await screen.findByText('Unable to check training jobs')).toBeVisible();
            expect(screen.getByText('Training already queued')).toBeVisible();
        });

        it.each([['running'], ['active']] as const)(
            'shows the device conflict warning when pipeline is %s on the same device',
            async (status) => {
                renderDialog({
                    devices: CPU_ONLY,
                    pipelineConfig: { status, inference_device: 'CPU' },
                });

                expect(await screen.findByText(/Inference pipeline is running on CPU/i)).toBeVisible();
            }
        );

        it('does not show the device conflict warning when pipeline is idle', async () => {
            renderDialog({ devices: CPU_ONLY, pipelineConfig: { status: 'idle', inference_device: 'CPU' } });

            expect(await screen.findByText('Train model')).toBeVisible();
            expect(screen.queryByText(/Inference pipeline is (?:active|running) on/i)).not.toBeInTheDocument();
        });

        it('does not show the device conflict warning when pipeline runs on a different device', async () => {
            // XPU is the preferred default device; CPU pipeline should not conflict
            renderDialog({
                devices: [
                    { type: 'cpu', name: 'CPU', memory: null, index: null },
                    { type: 'xpu', name: 'Intel(R) Graphics', memory: 30673268736, index: 0 },
                ],
                pipelineConfig: { status: 'running', inference_device: 'CPU' },
            });

            expect(await screen.findByText('Train model')).toBeVisible();
            expect(screen.queryByText(/Inference pipeline is (?:active|running) on/i)).not.toBeInTheDocument();
        });

        it('clears the device conflict warning when the user switches to a non-conflicting device', async () => {
            renderDialog({
                devices: [
                    { type: 'cpu', name: 'CPU', memory: null, index: null },
                    { type: 'xpu', name: 'Intel(R) Graphics', memory: 30673268736, index: 0 },
                ],
                pipelineConfig: { status: 'running', inference_device: 'CPU' },
            });

            await userEvent.click(await screen.findByRole('button', { name: /select a training device/i }));
            await userEvent.click(await screen.findByRole('option', { name: /^CPU$/i }));
            expect(await screen.findByText(/Inference pipeline is running on CPU/i)).toBeVisible();

            await userEvent.click(screen.getByRole('button', { name: /select a training device/i }));
            await userEvent.click(await screen.findByRole('option', { name: /Intel\(R\) Graphics/i }));
            expect(screen.queryByText(/Inference pipeline is (?:active|running) on/i)).not.toBeInTheDocument();
        });
    });
});
