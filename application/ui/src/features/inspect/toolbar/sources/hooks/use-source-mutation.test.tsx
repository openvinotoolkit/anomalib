// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { act } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { renderHook } from '../../../../../../tests/utils';
import { UsbCameraSourceConfig } from '../util';
import { useSourceMutation } from './use-source-mutation.hook';

const mockedSource: UsbCameraSourceConfig = {
    id: 'original-id',
    project_id: '123',
    name: 'Mock Source',
    source_type: 'usb_camera' as const,
    device_id: 0,
};

describe('useSourceMutation', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('creates a new source and return its resource id', async () => {
        const { result } = renderHook(() => useSourceMutation(true), {
            route: '/projects/project-id-123/inspect',
        });

        const createdSource = { ...mockedSource, id: 'created-id' };
        server.use(
            http.post('/api/projects/{project_id}/sources', async () => {
                return HttpResponse.json(createdSource);
            }),
            http.patch('/api/projects/{project_id}/sources/{source_id}', () => HttpResponse.error())
        );

        await act(async () => {
            const response = await result.current(mockedSource);
            expect(response).toBe(createdSource.id);
        });
    });

    it('update a source item and returns its resource id', async () => {
        const { result } = renderHook(() => useSourceMutation(false), {
            route: '/projects/project-id-123/inspect',
        });

        server.use(
            http.post('/api/projects/{project_id}/sources', () => HttpResponse.error()),
            http.patch('/api/projects/{project_id}/sources/{source_id}', () => HttpResponse.json(mockedSource))
        );

        await act(async () => {
            const response = await result.current(mockedSource);
            expect(response).toBe(mockedSource.id);
        });
    });
});
