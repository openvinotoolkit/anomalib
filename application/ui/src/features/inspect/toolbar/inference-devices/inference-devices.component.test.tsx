// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@anomalib-studio/toast';
import { ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { InferenceDevices } from './inference-devices.component';

vi.mock('@anomalib-studio/toast', async () => {
    const actual = await vi.importActual('@anomalib-studio/toast');
    return { ...actual, toast: vi.fn() };
});

type DeviceType = 'cpu' | 'xpu' | 'cuda';

interface MockDeviceInfo {
    type: DeviceType;
    name: string;
    memory: number | null;
    index: number | null;
    openvino_name: string | null;
}

const INFERENCE_DEVICES: MockDeviceInfo[] = [
    { type: 'cpu', name: 'CPU', memory: null, index: null, openvino_name: 'CPU' },
    { type: 'xpu', name: 'Intel(R) Graphics', memory: 30673268736, index: 0, openvino_name: 'GPU.0' },
];

const TRAINING_DEVICES_WITH_CUDA: MockDeviceInfo[] = [
    { type: 'cpu', name: 'CPU', memory: null, index: null, openvino_name: null },
    { type: 'cuda', name: 'NVIDIA RTX 4090', memory: 8589934592, index: 0, openvino_name: null },
];

const TRAINING_DEVICES_NO_CUDA: MockDeviceInfo[] = [
    { type: 'cpu', name: 'CPU', memory: null, index: null, openvino_name: null },
    { type: 'xpu', name: 'Intel(R) Graphics', memory: 30673268736, index: 0, openvino_name: null },
];

describe('InferenceDevices', () => {
    const renderApp = ({
        inferenceDevices = INFERENCE_DEVICES,
        trainingDevices = TRAINING_DEVICES_NO_CUDA,
        pipelineConfig = {},
    }: {
        inferenceDevices?: MockDeviceInfo[];
        trainingDevices?: MockDeviceInfo[];
        pipelineConfig?: Partial<SchemaPipeline>;
    } = {}) => {
        server.use(
            http.get('/api/system/devices/inference', ({ response }) => response(200).json(inferenceDevices)),
            http.get('/api/system/devices/training', ({ response }) => response(200).json(trainingDevices)),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
            )
        );

        return render(
            <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
                <ThemeProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect']}>
                        <Routes>
                            <Route path='/projects/:projectId/inspect' element={<InferenceDevices />} />
                        </Routes>
                    </MemoryRouter>
                </ThemeProvider>
            </QueryClientProvider>
        );
    };

    describe('Rendering', () => {
        it('renders the picker with inference device labels', async () => {
            renderApp();

            expect(await screen.findByRole('button', { name: /inference devices/i })).toBeVisible();
            // Picker renders items in both a hidden listbox and visible — use getAllByText
            expect(screen.getAllByText('CPU').length).toBeGreaterThanOrEqual(1);
            expect(screen.getAllByText('Intel(R) Graphics [0]').length).toBeGreaterThanOrEqual(1);
        });

        it('derives selected key from pipeline inference_device', async () => {
            renderApp({ pipelineConfig: { inference_device: 'GPU.0' } });

            // The picker button text should show the matching device label
            const picker = await screen.findByRole('button', { name: /inference devices/i });
            expect(picker).toHaveTextContent('Intel(R) Graphics [0]');
        });

        it('selects CPU when pipeline inference_device is CPU', async () => {
            renderApp({ pipelineConfig: { inference_device: 'CPU' } });

            const picker = await screen.findByRole('button', { name: /inference devices/i });
            expect(picker).toHaveTextContent('CPU');
        });
    });

    describe('Selection change', () => {
        it('calls PATCH with the selected device openvino_name', async () => {
            const requestSpy = vi.fn();

            server.use(
                http.patch('/api/projects/{project_id}/pipeline', ({ response }) => {
                    requestSpy();
                    return response(200).json(getMockedPipeline({ inference_device: 'GPU.0' }));
                })
            );

            renderApp({ pipelineConfig: { inference_device: 'CPU' } });

            const picker = await screen.findByRole('button', { name: /inference devices/i });
            fireEvent.click(picker);

            const option = await screen.findByRole('option', { name: /Intel\(R\) Graphics \[0\]/i });
            fireEvent.click(option);

            await waitFor(() => {
                expect(requestSpy).toHaveBeenCalled();
            });
        });

        it('shows error toast on mutation failure', async () => {
            server.use(
                http.patch('/api/projects/{project_id}/pipeline', () =>
                    HttpResponse.json({ detail: 'Internal server error' } as Record<string, unknown>, { status: 500 })
                )
            );

            renderApp({ pipelineConfig: { inference_device: 'CPU' } });

            const picker = await screen.findByRole('button', { name: /inference devices/i });
            fireEvent.click(picker);

            const option = await screen.findByRole('option', { name: /Intel\(R\) Graphics \[0\]/i });
            fireEvent.click(option);

            await waitFor(() => {
                expect(toast).toHaveBeenCalledWith(
                    expect.objectContaining({
                        type: 'error',
                        message: expect.stringContaining('Internal server error'),
                    })
                );
            });
        });
    });

    describe('NVIDIA notice', () => {
        it('shows contextual help when CUDA training devices exist', async () => {
            renderApp({ trainingDevices: TRAINING_DEVICES_WITH_CUDA });

            // The ContextualHelp renders an info button
            const infoButton = await screen.findByRole('button', { name: /info/i });
            expect(infoButton).toBeVisible();

            await userEvent.click(infoButton);

            expect(await screen.findByText('NVIDIA GPUs')).toBeVisible();
            expect(
                screen.getByText(/NVIDIA GPUs are available for training but are not supported for inference/i)
            ).toBeVisible();
        });

        it('does not show contextual help when no CUDA training devices', async () => {
            renderApp({ trainingDevices: TRAINING_DEVICES_NO_CUDA });

            await screen.findByRole('button', { name: /inference devices/i });

            expect(screen.queryByRole('button', { name: /info/i })).not.toBeInTheDocument();
        });
    });
});
