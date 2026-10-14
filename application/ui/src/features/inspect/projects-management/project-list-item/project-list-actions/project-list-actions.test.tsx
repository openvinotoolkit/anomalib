import { ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes } from 'react-router';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { vi } from 'vitest';

import { ProjectActions } from './project-list-actions.component';

const renderWithProviders = (ui: React.ReactElement, { route = '/projects/project-123/inspect' } = {}) => {
    return render(
        <QueryClientProvider client={new QueryClient()}>
            <ThemeProvider>
                <MemoryRouter initialEntries={[route]}>
                    <Routes>
                        <Route path='/projects/:projectId/inspect' element={ui} />
                    </Routes>
                </MemoryRouter>
            </ThemeProvider>
        </QueryClientProvider>
    );
};

describe('ProjectActions', () => {
    const defaultProps = {
        projectId: 'project-456',
        projectName: 'Test Project',
        isLastProject: false,
        onRename: vi.fn(),
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('displays rename option in the menu', async () => {
        renderWithProviders(<ProjectActions {...defaultProps} />);

        await userEvent.click(screen.getByRole('button', { name: /project actions/i }));

        expect(screen.getByRole('menuitem', { name: /Rename/i })).toBeVisible();
    });

    it('displays delete option in the menu', async () => {
        renderWithProviders(<ProjectActions {...defaultProps} />);

        await userEvent.click(screen.getByRole('button', { name: /project actions/i }));

        expect(screen.getByRole('menuitem', { name: /Delete/i })).toBeVisible();
    });

    it('calls onRename when rename menu item is clicked', async () => {
        const mockOnRename = vi.fn();
        renderWithProviders(<ProjectActions {...defaultProps} onRename={mockOnRename} />);

        await userEvent.click(screen.getByRole('button', { name: /project actions/i }));
        await userEvent.click(screen.getByRole('menuitem', { name: /Rename/i }));

        expect(mockOnRename).toHaveBeenCalledTimes(1);
    });

    it('opens delete confirmation dialog when delete menu item is clicked', async () => {
        renderWithProviders(<ProjectActions {...defaultProps} />);

        await userEvent.click(screen.getByRole('button', { name: /project actions/i }));
        await userEvent.click(screen.getByRole('menuitem', { name: /Delete/i }));

        expect(screen.getByRole('alertdialog')).toBeVisible();
        expect(screen.getByText(/Delete project "Test Project"\?/i)).toBeVisible();
    });

    it('disables delete option when project is the last project', async () => {
        renderWithProviders(<ProjectActions {...defaultProps} isLastProject={true} />);

        await userEvent.click(screen.getByRole('button', { name: /project actions/i }));

        const deleteMenuItem = screen.getByRole('menuitem', { name: /Delete/i });
        expect(deleteMenuItem).toHaveAttribute('aria-disabled', 'true');
    });

    it('disables delete option when project is the currently active project', async () => {
        renderWithProviders(<ProjectActions {...defaultProps} projectId='project-123' />, {
            route: '/projects/project-123/inspect',
        });

        await userEvent.click(screen.getByRole('button', { name: /project actions/i }));

        const deleteMenuItem = screen.getByRole('menuitem', { name: /Delete/i });
        expect(deleteMenuItem).toHaveAttribute('aria-disabled', 'true');
    });

    it('calls delete API when delete is confirmed', async () => {
        const deleteSpy = vi.fn();
        server.use(
            http.delete('/api/projects/{project_id}', () => {
                deleteSpy();
                return HttpResponse.json({});
            })
        );

        renderWithProviders(<ProjectActions {...defaultProps} />);

        await userEvent.click(screen.getByRole('button', { name: /project actions/i }));
        await userEvent.click(screen.getByRole('menuitem', { name: /Delete/i }));
        await userEvent.click(screen.getByRole('button', { name: /^Delete$/i }));

        await waitFor(() => {
            expect(deleteSpy).toHaveBeenCalled();
        });
    });

    it('closes delete dialog when cancel is clicked', async () => {
        renderWithProviders(<ProjectActions {...defaultProps} />);

        await userEvent.click(screen.getByRole('button', { name: /project actions/i }));
        await userEvent.click(screen.getByRole('menuitem', { name: /Delete/i }));

        expect(screen.getByRole('alertdialog')).toBeVisible();

        await userEvent.click(screen.getByRole('button', { name: /Cancel/i }));

        await waitFor(() => {
            expect(screen.queryByRole('alertdialog')).not.toBeInTheDocument();
        });
    });
});
