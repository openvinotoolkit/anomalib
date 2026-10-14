// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DialogContainer, ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TRAINABLE_MODELS } from 'mocks/mock-trainable-models';
import { MemoryRouter, Route, Routes } from 'react-router';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

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

describe('TrainModelDialog', () => {
    const closeMock = vi.fn();

    const renderDialog = ({ devices = DEFAULT_MOCK_DEVICES }: { devices?: MockDeviceInfo[] } = {}) => {
        server.use(
            http.get('/api/trainable-models', ({ response }) =>
                response(200).json({ trainable_models: TRAINABLE_MODELS })
            ),
            http.get('/api/system/devices/training', ({ response }) => response(200).json(devices)),
            http.post('/api/jobs:train', ({ response }) => response(200).json({ job_id: 'job-123' }))
        );

        return render(
            <QueryClientProvider client={new QueryClient()}>
                <ThemeProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect']}>
                        <Routes>
                            <Route
                                path='/projects/:projectId/inspect'
                                element={
                                    <DialogContainer onDismiss={closeMock}>
                                        <TrainModelDialog close={closeMock} />
                                    </DialogContainer>
                                }
                            />
                        </Routes>
                    </MemoryRouter>
                </ThemeProvider>
            </QueryClientProvider>
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
});
