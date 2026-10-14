// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { queryClient } from 'src/query-client/query-client';

import { renderHook } from '../../../../../tests/utils';
import { useActivePipelineStatus } from './use-active-pipeline-status.hook';

describe('useActivePipelineStatus', () => {
    const mockProjectId = 'project-id-123';

    const renderHookWithProviders = (projectId: string) => renderHook(() => useActivePipelineStatus(projectId));

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    describe('Active Pipeline Detection', () => {
        it('returns hasActiveProject as true when there is an active pipeline', async () => {
            server.use(
                http.get('/api/active-pipeline', () =>
                    HttpResponse.json({ project_id: mockProjectId, status: 'idle', inference_device: 'CPU' })
                )
            );

            const { result } = renderHookWithProviders('123');

            await waitFor(() => {
                expect(result.current?.hasActiveProject).toBe(true);
            });
        });

        it('returns hasActiveProject as false when there is no active pipeline', async () => {
            server.use(http.get('/api/active-pipeline', () => HttpResponse.json()));

            const { result } = renderHookWithProviders('123');

            await waitFor(() => {
                expect(result.current.hasActiveProject).toBe(false);
            });
        });

        it('returns correct activeProjectId when pipeline is active', async () => {
            const activeProjectId = '789';
            const currentProjectId = '321';
            server.use(
                http.get('/api/active-pipeline', () =>
                    HttpResponse.json({ project_id: activeProjectId, status: 'idle', inference_device: 'CPU' })
                )
            );

            const { result } = renderHookWithProviders(currentProjectId);

            await waitFor(() => {
                expect(result.current.isCurrentProjectActive).toBe(false);
                expect(result.current.activeProjectId).toBe(activeProjectId);
            });
        });
    });
});
