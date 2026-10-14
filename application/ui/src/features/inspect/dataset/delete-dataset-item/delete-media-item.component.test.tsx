// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router';

import { http } from '../../../../api/utils';
import { Toast } from '../../../../components/toast/toast.component';
import { server } from '../../../../msw-node-setup';
import { DeleteMediaItem, DeleteMediaItemProps } from './delete-dataset-item.component';

describe('DeleteMediaItem', () => {
    const renderApp = (props: DeleteMediaItemProps) => {
        return render(
            <QueryClientProvider client={new QueryClient()}>
                <ThemeProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect']}>
                        <Routes>
                            <Route path='/projects/:projectId/inspect' element={<DeleteMediaItem {...props} />} />
                        </Routes>
                    </MemoryRouter>
                    <Toast />
                </ThemeProvider>
            </QueryClientProvider>
        );
    };

    it('deletes a media item and shows a success toast', async () => {
        const itemId = '123';
        const mockedOnDeleted = vitest.fn();

        server.use(
            http.delete('/api/projects/{project_id}/images/{media_id}', () => {
                return HttpResponse.json(null, { status: 204 });
            })
        );

        renderApp({ itemsIds: [itemId], onDeleted: mockedOnDeleted });

        fireEvent.click(screen.getByLabelText(/delete media item/i));
        await screen.findByText(/Are you sure you want to delete 1 item(s)?/i);

        fireEvent.click(screen.getByRole('button', { name: /confirm/i }));

        expect(await screen.findByText(`1 item(s) deleted successfully`)).toBeVisible();
        expect(mockedOnDeleted).toHaveBeenCalledWith([itemId]);
    });

    it('shows an error toast when deleting a media item fails', async () => {
        const itemToFail = '321';
        const itemToDelete = '123';
        const errorMessage = 'test error message';
        const mockedOnDeleted = vitest.fn();

        server.use(
            http.delete('/api/projects/{project_id}/images/{media_id}', ({ params }) => {
                const { media_id } = params;
                return media_id === itemToDelete
                    ? HttpResponse.json(null, { status: 204 })
                    : // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                      // @ts-expect-error
                      HttpResponse.json({ detail: errorMessage }, { status: 500 });
            })
        );

        renderApp({ itemsIds: [itemToFail, itemToDelete], onDeleted: mockedOnDeleted });

        fireEvent.click(screen.getByLabelText(/delete media item/i));
        await screen.findByText(/Are you sure you want to delete 2 item(s)?/i);

        fireEvent.click(screen.getByRole('button', { name: /confirm/i }));

        expect(await screen.findByText(`1 item(s) deleted successfully`)).toBeVisible();
        expect(await screen.findByText(`Failed to delete, ${errorMessage}`)).toBeVisible();
        expect(mockedOnDeleted).toHaveBeenCalledWith([itemToDelete]);
    });
});
