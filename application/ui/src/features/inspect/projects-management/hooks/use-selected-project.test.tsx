// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { waitFor } from '@testing-library/react';
import { getMockedProject } from 'mocks/mock-project';

import { renderHook } from '../../../../../tests/utils';
import { useGetProjects } from './use-get-project.hooks';
import { useSelectedProject } from './use-selected-project.hook';

vi.mock('./use-get-project.hooks', () => ({
    useGetProjects: vi.fn(),
}));

describe('useSelectedProject', () => {
    const mockProjects = [
        getMockedProject({ id: 'project-1', name: 'Project 1' }),
        getMockedProject({ id: 'project-2', name: 'Project 2' }),
        getMockedProject({ id: 'project-3', name: 'Project 3' }),
    ];

    const renderApp = (props: Partial<ReturnType<typeof useGetProjects>> & { projectId?: string }) => {
        vi.mocked(useGetProjects).mockReturnValue({
            isLoading: false,
            hasNextPage: false,
            projects: mockProjects,
            isFetchingNextPage: false,
            fetchNextPage: vi.fn(),
            ...props,
        });

        return renderHook(() => useSelectedProject(), {
            route: `/projects/${props.projectId ?? 'project-123'}/inspect`,
        });
    };

    it('uses the project id from the route', () => {
        const { result } = renderApp({
            projects: mockProjects,
            projectId: 'project-3',
        });

        expect(result.current).toEqual(mockProjects[2]);
    });

    it('does not fetch next page when selectedProject is found', () => {
        const mockFetchNextPage = vi.fn();
        const projectId = mockProjects[0].id;

        renderApp({
            hasNextPage: true,
            fetchNextPage: mockFetchNextPage,
            projects: mockProjects,
            projectId,
        });

        expect(mockFetchNextPage).not.toHaveBeenCalled();
    });

    it('fetches next page when selectedProject is not found and hasNextPage is true', async () => {
        const mockFetchNextPage = vi.fn();

        renderApp({ hasNextPage: true, fetchNextPage: mockFetchNextPage, projectId: 'project-999' });

        await waitFor(() => {
            expect(mockFetchNextPage).toHaveBeenCalled();
        });
    });

    it('does not fetch next page when isFetchingNextPage is true', () => {
        const mockFetchNextPage = vi.fn();

        renderApp({
            hasNextPage: true,
            isFetchingNextPage: true,
            projectId: 'project-999',
            fetchNextPage: mockFetchNextPage,
        });

        expect(mockFetchNextPage).not.toHaveBeenCalled();
    });

    it('does not fetch next page when hasNextPage is false', () => {
        const mockFetchNextPage = vi.fn();

        renderApp({ hasNextPage: false, fetchNextPage: mockFetchNextPage, projectId: 'project-999' });

        expect(mockFetchNextPage).not.toHaveBeenCalled();
    });

    it('returns the selected project when found', () => {
        const projectId = mockProjects[1].id;

        const { result } = renderApp({
            projects: mockProjects,
            projectId,
        });

        expect(result.current).toEqual(mockProjects[1]);
    });

    it('returns undefined when selectedProject is not found', () => {
        const { result } = renderApp({
            projects: mockProjects,
            projectId: 'project-999',
            hasNextPage: false,
        });

        expect(result.current).toBeUndefined();
    });
});
