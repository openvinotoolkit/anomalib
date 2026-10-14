import { Size, ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { getMockedMediaItem } from '../../../../../../mocks/mock-media-item';
import { Toast } from '../../../../../components/toast/toast.component';
import { MediaItem } from '../../types';
import { SidebarItems } from './sidebar-items.component';

const mockMediaItems: MediaItem[] = [
    getMockedMediaItem({ id: '1', filename: 'image1.jpg', project_id: '123' }),
    getMockedMediaItem({ id: '2', filename: 'image2.jpg', project_id: '123' }),
    getMockedMediaItem({ id: '3', filename: 'image3.jpg', project_id: '123' }),
];

describe('SidebarItems', () => {
    const renderApp = ({
        mediaItems,
        selectedMediaItem,
        onSelectedMediaItem = vi.fn(),
    }: {
        mediaItems: MediaItem[];
        selectedMediaItem: MediaItem;
        onSelectedMediaItem?: (mediaItem: string | null) => Promise<URLSearchParams>;
    }) => {
        server.use(
            http.get('/api/projects/{project_id}/images/{media_id}/thumbnail', () =>
                HttpResponse.arrayBuffer(new ArrayBuffer(0), {
                    headers: { 'Content-Type': 'image/jpeg' },
                })
            )
        );

        return render(
            <QueryClientProvider client={new QueryClient()}>
                <ThemeProvider>
                    <MemoryRouter initialEntries={['/projects/123/inspect']}>
                        <Routes>
                            <Route
                                path='/projects/:projectId/inspect'
                                element={
                                    <SidebarItems
                                        layoutOptions={{
                                            maxColumns: 1,
                                            maxItemSize: new Size(100, 100),
                                            maxHorizontalSpace: 1000,
                                        }}
                                        mediaItems={mediaItems}
                                        selectedMediaItem={selectedMediaItem}
                                        onSelectedMediaItem={onSelectedMediaItem}
                                        hasNextPage={false}
                                        isLoadingMore={false}
                                        loadMore={vi.fn()}
                                    />
                                }
                            />
                        </Routes>
                    </MemoryRouter>
                    <Toast />
                </ThemeProvider>
            </QueryClientProvider>
        );
    };

    it('calls onSelectedMediaItem when thumbnail is clicked', async () => {
        const onSelectedMediaItem = vi.fn();
        const selectedMediaItem = mockMediaItems[0];

        renderApp({ mediaItems: mockMediaItems, selectedMediaItem, onSelectedMediaItem });

        const image = await screen.findByAltText(selectedMediaItem.filename);
        await userEvent.click(image);

        expect(onSelectedMediaItem).toHaveBeenCalledWith(selectedMediaItem.id);
    });

    describe('Deletion', () => {
        beforeEach(() => {
            server.use(
                http.delete('/api/projects/{project_id}/images/{media_id}', () => {
                    return HttpResponse.json(null, { status: 204 });
                })
            );
        });

        it('selects next item when selected item is deleted', async () => {
            const onSelectedMediaItem = vi.fn();
            const nextMediaItem = mockMediaItems[1];
            const selectedMediaItem = mockMediaItems[0];

            renderApp({ mediaItems: mockMediaItems, selectedMediaItem, onSelectedMediaItem });

            await screen.findByAltText(selectedMediaItem.filename);

            const deleteButtons = screen.getAllByLabelText(/delete media item/i);
            await userEvent.click(deleteButtons[0]);
            await userEvent.click(screen.getByRole('button', { name: /confirm/i }));

            await waitFor(() => {
                expect(onSelectedMediaItem).toHaveBeenCalledWith(nextMediaItem.id);
            });
        });

        it('selects previous item when last item is deleted', async () => {
            const onSelectedMediaItem = vi.fn();
            const nextMediaItem = mockMediaItems[1];
            const selectedMediaItem = mockMediaItems[2];

            renderApp({ mediaItems: mockMediaItems, selectedMediaItem, onSelectedMediaItem });

            await screen.findByAltText(selectedMediaItem.filename);

            const deleteButtons = screen.getAllByLabelText(/delete media item/i);
            await userEvent.click(deleteButtons[2]);

            await userEvent.click(screen.getByRole('button', { name: /confirm/i }));

            await waitFor(() => {
                expect(onSelectedMediaItem).toHaveBeenCalledWith(nextMediaItem.id);
            });
        });

        it('does not change selection when non-selected item is deleted', async () => {
            const onSelectedMediaItem = vi.fn();
            const selectedMediaItem = mockMediaItems[0];
            renderApp({ mediaItems: mockMediaItems, selectedMediaItem, onSelectedMediaItem });

            await screen.findByAltText(selectedMediaItem.filename);

            const deleteButtons = screen.getAllByLabelText(/delete media item/i);
            await userEvent.click(deleteButtons[1]);

            await userEvent.click(screen.getByRole('button', { name: /confirm/i }));

            await waitFor(() => {
                expect(screen.getByText(/1 item\(s\) deleted successfully/i)).toBeVisible();
            });

            expect(onSelectedMediaItem).not.toHaveBeenCalled();
        });

        it('calls onSelectedMediaItem with null when only item is deleted', async () => {
            const onSelectedMediaItem = vi.fn();
            const singleItem = [getMockedMediaItem({ id: '1', filename: 'single.jpg', project_id: '123' })];

            renderApp({ mediaItems: singleItem, selectedMediaItem: singleItem[0], onSelectedMediaItem });

            await screen.findByAltText('single.jpg');

            const deleteButton = screen.getByLabelText(/delete media item/i);
            await userEvent.click(deleteButton);

            await screen.findByText(/Are you sure you want to delete 1 item\(s\)?/i);
            await userEvent.click(screen.getByRole('button', { name: /confirm/i }));

            await waitFor(() => {
                expect(onSelectedMediaItem).toHaveBeenCalledWith(null);
            });
        });

        it('selects previous item when selected middle item is deleted', async () => {
            const onSelectedMediaItem = vi.fn();
            const prevMediaItem = mockMediaItems[0];
            const selectedMediaItem = mockMediaItems[1];

            renderApp({ mediaItems: mockMediaItems, selectedMediaItem, onSelectedMediaItem });

            await screen.findByAltText('image2.jpg');

            const deleteButtons = screen.getAllByLabelText(/delete media item/i);
            await userEvent.click(deleteButtons[1]);

            await screen.findByText(/Are you sure you want to delete 1 item\(s\)?/i);
            await userEvent.click(screen.getByRole('button', { name: /confirm/i }));

            await waitFor(() => {
                expect(onSelectedMediaItem).toHaveBeenCalledWith(prevMediaItem.id);
            });
        });

        it('shows error toast when delete fails', async () => {
            const errorMessage = 'Failed to delete image';

            server.use(
                http.delete('/api/projects/{project_id}/images/{media_id}', () => {
                    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                    // @ts-expect-error
                    return HttpResponse.json({ detail: errorMessage }, { status: 500 });
                })
            );

            renderApp({ mediaItems: mockMediaItems, selectedMediaItem: mockMediaItems[0] });

            await screen.findByAltText('image1.jpg');

            const deleteButtons = screen.getAllByLabelText(/delete media item/i);
            await userEvent.click(deleteButtons[0]);

            await screen.findByText(/Are you sure you want to delete 1 item\(s\)?/i);
            await userEvent.click(screen.getByRole('button', { name: /confirm/i }));

            expect(await screen.findByText(`Failed to delete, ${errorMessage}`)).toBeVisible();
        });
    });

    describe('Edge Cases', () => {
        it('handles empty media items array', () => {
            renderApp({ mediaItems: [], selectedMediaItem: mockMediaItems[0] });

            expect(screen.queryByRole('img')).not.toBeInTheDocument();
        });

        it('handles selected item not in media items list', async () => {
            const selectedItem = getMockedMediaItem({ id: '999', filename: 'missing.jpg', project_id: '123' });
            renderApp({ mediaItems: mockMediaItems, selectedMediaItem: selectedItem });

            expect(await screen.findByAltText('image1.jpg')).toBeInTheDocument();
        });
    });
});
