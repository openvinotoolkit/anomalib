// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { DialogContainer } from '@geti-ui/ui';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { getMockedProject } from 'mocks/mock-project';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { vi } from 'vitest';

import { render } from '../../../../../tests/utils';
import { ConfirmationDialog } from './confirmation-dialog.component';

const ACTIVE_PROJECT_ID = 'active-project-id';
const CURRENT_PROJECT_ID = 'current-project-id';

const renderConfirmationDialog = () => {
    const Harness = () => {
        const [isOpen, setIsOpen] = useState(true);
        return (
            <DialogContainer onDismiss={() => setIsOpen(false)}>
                {isOpen && (
                    <Suspense fallback={<div>Loading...</div>}>
                        <ConfirmationDialog activeProjectId={ACTIVE_PROJECT_ID} currentProjectId={CURRENT_PROJECT_ID} />
                    </Suspense>
                )}
            </DialogContainer>
        );
    };

    return render(<Harness />, { route: `/projects/${CURRENT_PROJECT_ID}/inspect` });
};

describe('ConfirmationDialog', () => {
    beforeEach(() => {
        server.use(
            http.get('/api/projects/{project_id}', ({ params }) => {
                if (params.project_id === ACTIVE_PROJECT_ID) {
                    return HttpResponse.json(getMockedProject({ id: ACTIVE_PROJECT_ID, name: 'Active Project' }));
                }
                return HttpResponse.json(getMockedProject({ id: CURRENT_PROJECT_ID, name: 'Current Project' }));
            }),
            http.get('/api/projects/{project_id}/pipeline', () => HttpResponse.json(getMockedPipeline()))
        );
    });

    it('shows the current and active project names in the confirmation copy', async () => {
        renderConfirmationDialog();

        expect(await screen.findByRole('alertdialog')).toBeVisible();
        expect(screen.getByRole('heading', { name: /Activate project "Current Project"/i })).toBeVisible();
        expect(screen.getByText(/current active project "Active Project"/i)).toBeVisible();
    });

    it('disables the active pipeline and activates+runs the current pipeline when it is fully configured', async () => {
        const disableSpy = vi.fn();
        const activateSpy = vi.fn();
        const runSpy = vi.fn();
        const callOrder: string[] = [];

        server.use(
            http.post('/api/projects/{project_id}/pipeline:disable', ({ params }) => {
                disableSpy(params.project_id);
                callOrder.push('disable');
                return HttpResponse.json({});
            }),
            http.post('/api/projects/{project_id}/pipeline:activate', ({ params }) => {
                activateSpy(params.project_id);
                callOrder.push('activate');
                return HttpResponse.json({});
            }),
            http.post('/api/projects/{project_id}/pipeline:run', ({ params }) => {
                runSpy(params.project_id);
                callOrder.push('run');
                return HttpResponse.json({});
            })
        );

        renderConfirmationDialog();

        await userEvent.click(await screen.findByRole('button', { name: /^Activate project$/i }));

        await waitFor(() => {
            expect(disableSpy).toHaveBeenCalledWith(ACTIVE_PROJECT_ID);
            expect(activateSpy).toHaveBeenCalledWith(CURRENT_PROJECT_ID);
            expect(runSpy).toHaveBeenCalledWith(CURRENT_PROJECT_ID);
        });

        expect(callOrder).toEqual(['disable', 'activate', 'run']);
    });

    it('activates without running when the current pipeline is not fully configured', async () => {
        const disableSpy = vi.fn();
        const activateSpy = vi.fn();
        const runSpy = vi.fn();

        server.use(
            http.get('/api/projects/{project_id}/pipeline', () =>
                HttpResponse.json(getMockedPipeline({ source: undefined, model: undefined }))
            ),
            http.post('/api/projects/{project_id}/pipeline:disable', ({ params }) => {
                disableSpy(params.project_id);
                return HttpResponse.json({});
            }),
            http.post('/api/projects/{project_id}/pipeline:activate', ({ params }) => {
                activateSpy(params.project_id);
                return HttpResponse.json({});
            }),
            http.post('/api/projects/{project_id}/pipeline:run', ({ params }) => {
                runSpy(params.project_id);
                return HttpResponse.json({});
            })
        );

        renderConfirmationDialog();

        await userEvent.click(await screen.findByRole('button', { name: /^Activate project$/i }));

        await waitFor(() => {
            expect(disableSpy).toHaveBeenCalledWith(ACTIVE_PROJECT_ID);
            expect(activateSpy).toHaveBeenCalledWith(CURRENT_PROJECT_ID);
        });
        expect(runSpy).not.toHaveBeenCalled();
    });

    it('closes without invoking any pipeline mutation when cancel is clicked', async () => {
        const disableSpy = vi.fn();
        const activateSpy = vi.fn();
        const runSpy = vi.fn();

        server.use(
            http.post('/api/projects/{project_id}/pipeline:disable', () => {
                disableSpy();
                return HttpResponse.json({});
            }),
            http.post('/api/projects/{project_id}/pipeline:activate', () => {
                activateSpy();
                return HttpResponse.json({});
            }),
            http.post('/api/projects/{project_id}/pipeline:run', () => {
                runSpy();
                return HttpResponse.json({});
            })
        );

        renderConfirmationDialog();

        expect(await screen.findByRole('alertdialog')).toBeVisible();
        await userEvent.click(screen.getByRole('button', { name: /^Cancel$/i }));

        await waitFor(() => {
            expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
        });
        expect(disableSpy).not.toHaveBeenCalled();
        expect(activateSpy).not.toHaveBeenCalled();
        expect(runSpy).not.toHaveBeenCalled();
    });
});
