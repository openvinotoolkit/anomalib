import { $api } from '@anomalib-studio/api';

const projectsItemsLimit = 20;

export const useGetProjects = () => {
    const { data, isLoading, fetchNextPage, hasNextPage, isFetchingNextPage } = $api.useInfiniteQuery(
        'get',
        '/api/projects',
        {
            params: {
                query: { offset: 0, limit: projectsItemsLimit },
            },
        },
        {
            pageParamName: 'offset',
            getNextPageParam: ({
                pagination,
            }: {
                pagination: { offset: number; limit: number; count: number; total: number };
            }) => {
                const total = pagination.offset + pagination.count;

                if (total >= pagination.total) {
                    return undefined;
                }

                return pagination.offset + projectsItemsLimit;
            },
            initialPageParam: 0,
        }
    );

    const projects = data?.pages.flatMap((page) => page.projects) ?? [];

    return { projects, isLoading, fetchNextPage, hasNextPage, isFetchingNextPage };
};
