import { ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedMediaItem } from 'mocks/mock-media-item';
import { HttpResponse } from 'msw';
import { MemoryRouter, Route, Routes, useNavigate } from 'react-router';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { vi } from 'vitest';

import { useStreamConnection } from '../../../../components/stream/stream-connection-provider';
import { Project, ProjectListItem } from './project-list-item.component';

vi.mock('../../../../components/stream/stream-connection-provider');

vi.mock('react-router', async () => {
    const actual = await vi.importActual('react-router');
    return {
        ...actual,
        useNavigate: vi.fn(),
    };
});

const mockNavigate = vi.fn();

const renderWithRouting = (ui: React.ReactElement, { route = '/projects/current-project' } = {}) => {
    return render(
        <QueryClientProvider client={new QueryClient()}>
            <ThemeProvider>
                <MemoryRouter initialEntries={[route]}>
                    <Routes>
                        <Route path='/projects/:projectId' element={ui} />
                    </Routes>
                </MemoryRouter>
            </ThemeProvider>
        </QueryClientProvider>
    );
};

describe('ProjectListItem', () => {
    const mockProject: Project = {
        id: 'project-123',
        name: 'Test Project',
    };

    beforeEach(() => {
        vi.clearAllMocks();
        vi.mocked(useNavigate).mockReturnValue(mockNavigate);
        vi.mocked(useStreamConnection).mockReturnValue({
            stop: vi.fn(),
            start: vi.fn(),
            status: 'idle',
            streamUrl: null,
            setStatus: vi.fn(),
        });

        server.use(
            http.get('/api/projects/{project_id}/images', () => {
                return HttpResponse.json({
                    media: [],
                    pagination: { offset: 0, limit: 1, count: 0, total: 0 },
                });
            })
        );
    });

    it('navigates to project when clicked', async () => {
        renderWithRouting(
            <ProjectListItem
                project={mockProject}
                isInEditMode={false}
                isActive={false}
                isLastProject={false}
                setProjectInEdition={vi.fn()}
            />
        );

        await userEvent.click(screen.getByRole('listitem'));

        expect(mockNavigate).toHaveBeenCalledWith('/projects/project-123?mode=Dataset');
    });

    it('updates project name when edited', async () => {
        const updateNameSpy = vi.fn();
        const mockedSetProjectInEdition = vi.fn();

        server.use(
            http.patch('/api/projects/{project_id}', () => {
                updateNameSpy();
                return HttpResponse.json({});
            })
        );

        renderWithRouting(
            <ProjectListItem
                project={mockProject}
                isActive={true}
                isInEditMode={true}
                isLastProject={false}
                setProjectInEdition={mockedSetProjectInEdition}
            />
        );

        const input = screen.getByRole('textbox', { name: /edit project name/i });
        await userEvent.clear(input);
        await userEvent.type(input, 'Updated Project Name');
        await userEvent.tab();

        expect(updateNameSpy).toHaveBeenCalled();
        expect(mockedSetProjectInEdition).toHaveBeenCalledWith(null);
    });

    it('displays project thumbnail when project has images', async () => {
        server.use(
            http.get('/api/projects/{project_id}/images', () => {
                return HttpResponse.json({
                    media: [getMockedMediaItem({ id: 'image-1', project_id: 'project-123' })],
                    pagination: { offset: 0, limit: 1, count: 1, total: 1 },
                });
            })
        );

        renderWithRouting(
            <ProjectListItem
                project={mockProject}
                isInEditMode={false}
                isActive={false}
                isLastProject={false}
                setProjectInEdition={vi.fn()}
            />
        );

        await waitFor(() => {
            expect(screen.getByRole('img', { name: /Test Project thumbnail/i })).toBeVisible();
        });
    });

    it('displays placeholder when project has no images', async () => {
        renderWithRouting(
            <ProjectListItem
                project={mockProject}
                isInEditMode={false}
                isActive={false}
                isLastProject={false}
                setProjectInEdition={vi.fn()}
            />
        );

        await waitFor(() => {
            expect(screen.queryByRole('img', { name: /thumbnail/i })).not.toBeInTheDocument();
        });
    });

    it('passes isLastProject prop to ProjectActions', async () => {
        renderWithRouting(
            <ProjectListItem
                project={mockProject}
                isInEditMode={false}
                isActive={false}
                isLastProject={true}
                setProjectInEdition={vi.fn()}
            />,
            { route: '/projects/other-project' }
        );

        await userEvent.click(screen.getByRole('button', { name: /project actions/i }));

        const deleteMenuItem = screen.getByRole('menuitem', { name: /Delete/i });
        expect(deleteMenuItem).toHaveAttribute('aria-disabled', 'true');
    });
});
