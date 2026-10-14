import { cleanup, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { toast as sonnerToast } from 'sonner';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { render } from '../../../../../../tests/utils';
import { SourceMenu, SourceMenuProps } from './source-menu.component';

describe('SourceMenu', () => {
    beforeEach(() => {
        cleanup();
    });

    const renderApp = ({
        id = 'id-test',
        name = 'name test',
        isConnected = false,
        onEdit = vi.fn(),
    }: Partial<SourceMenuProps>) => {
        render(<SourceMenu id={id} name={name} isConnected={isConnected} onEdit={onEdit} />, {
            route: '/projects/123/inspect',
        });
    };

    beforeEach(async () => {
        sonnerToast.dismiss();
    });

    it('edit', async () => {
        const mockedOnEdit = vi.fn();

        renderApp({ onEdit: mockedOnEdit });

        await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
        await userEvent.click(screen.getByRole('menuitem', { name: /Edit/i }));

        expect(mockedOnEdit).toHaveBeenCalled();
    });

    describe('remove', () => {
        const name = 'test-name';
        const configRequests = (status = 200) => {
            const pipelinePatchSpy = vi.fn();

            server.use(
                http.patch('/api/projects/{project_id}/pipeline', () => {
                    pipelinePatchSpy();
                    return HttpResponse.json({}, { status });
                }),
                http.delete('/api/projects/{project_id}/sources/{source_id}', () =>
                    HttpResponse.json(null, { status: 204 })
                )
            );

            return pipelinePatchSpy;
        };

        it('success', async () => {
            const pipelinePatchSpy = configRequests();

            renderApp({ name, isConnected: false });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Remove/i }));

            expect(await screen.findByText(`${name} has been removed successfully!`)).toBeInTheDocument();
            expect(pipelinePatchSpy).not.toHaveBeenCalled();
        });

        it('disabled when source is connected', async () => {
            renderApp({ name, isConnected: true });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));

            expect(screen.getByRole('menuitem', { name: /Remove/i })).toHaveAttribute('aria-disabled', 'true');
        });
    });

    describe('connect', () => {
        const name = 'test-name';
        const configRequests = (status = 200) => {
            server.use(http.patch('/api/projects/{project_id}/pipeline', () => HttpResponse.json({}, { status })));
        };

        it('success', async () => {
            configRequests();

            renderApp({ name });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Connect/i }));

            expect(await screen.findByText(`Successfully connected to "${name}"`)).toBeInTheDocument();
        });

        it('error', async () => {
            configRequests(500);

            renderApp({ name });

            await userEvent.click(screen.getByRole('button', { name: /source menu/i }));
            await userEvent.click(screen.getByRole('menuitem', { name: /Connect/i }));

            expect(await screen.findByText(`Failed to connect to "${name}".`)).toBeInTheDocument();
        });
    });
});
