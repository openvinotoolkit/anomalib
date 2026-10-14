// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { fetchClient } from '@anomalib-studio/api';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { getMockedModelData } from 'mocks/mock-model';
import { getMockedPagination } from 'mocks/mock-pagination';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { StatusBarProvider } from 'src/features/inspect/footer/status-bar/status-bar-context';
import { queryClient } from 'src/query-client/query-client';

import { render } from '../../../../../tests/utils';
import { ModelDetail } from './model-detail.component';

const projectId = 'test-project-id';

const mockedProject = getMockedProject();
const mockedPipeline = getMockedPipeline();

const renderModelDetail = (model: ReturnType<typeof getMockedModelData>, isActiveModel: boolean, onBack = vi.fn()) => {
    return render(
        <StatusBarProvider>
            <ModelDetail model={model} isActiveModel={isActiveModel} onBack={onBack} />
        </StatusBarProvider>,
        { queryClient, route: `/projects/${projectId}/inspect` }
    );
};

describe('ModelDetail', () => {
    beforeAll(() => {
        queryClient.setQueryData(
            ['get', '/api/projects/{project_id}', { params: { path: { project_id: projectId } } }],
            mockedProject
        );
        queryClient.setQueryData(['get', '/api/projects'], {
            projects: [mockedProject],
            ...getMockedPagination(),
        });
        queryClient.setQueryData(
            ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            mockedPipeline
        );
    });

    afterEach(() => {
        vi.clearAllMocks();
    });

    describe('Model Information', () => {
        it('displays model details and information grid', () => {
            const model = getMockedModelData();
            renderModelDetail(model, false);

            expect(screen.getAllByText('PatchCore').length).toBeGreaterThan(0);
            expect(screen.getByText('Training Date')).toBeInTheDocument();
            expect(screen.getByText('Model Size')).toBeInTheDocument();
            expect(screen.getByText('Training Duration')).toBeInTheDocument();
            expect(screen.getByText('Model Backbone')).toBeInTheDocument();
        });

        it('displays Active badge when isActiveModel is true', () => {
            const model = getMockedModelData();
            renderModelDetail(model, true);

            expect(screen.getByText('Active')).toBeInTheDocument();
        });

        it('does not display Active badge when isActiveModel is false', () => {
            const model = getMockedModelData();
            renderModelDetail(model, false);

            expect(screen.queryByText('Active')).not.toBeInTheDocument();
        });
    });

    describe('Navigation', () => {
        it('calls onBack when back button is clicked', () => {
            const model = getMockedModelData();
            const onBack = vi.fn();

            renderModelDetail(model, false, onBack);

            const backButton = screen.getByText('Back to Models');
            expect(backButton).toBeInTheDocument();
            fireEvent.click(backButton);
            expect(onBack).toHaveBeenCalledTimes(1);
        });
    });

    describe('Export Section', () => {
        it('calls export endpoint when Export button is clicked', async () => {
            const model = getMockedModelData();
            const exportSpy = vi
                .spyOn(fetchClient, 'POST')
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                .mockResolvedValue({ data: new Blob(), error: undefined } as any);

            renderModelDetail(model, false);

            expect(screen.getByRole('radiogroup', { name: 'Select export format' })).toBeInTheDocument();
            fireEvent.click(screen.getByLabelText('Select compression type'));
            fireEvent.click(screen.getByRole('option', { name: 'INT8' }));
            fireEvent.click(screen.getByRole('button', { name: 'Export' }));

            await waitFor(() => {
                expect(exportSpy).toHaveBeenCalledWith(
                    '/api/projects/{project_id}/models/{model_id}:export',
                    expect.objectContaining({
                        params: {
                            path: {
                                project_id: projectId,
                                model_id: model.id,
                            },
                        },
                        body: {
                            format: 'openvino',
                            compression: 'int8',
                        },
                    })
                );
            });
        });
    });
});
