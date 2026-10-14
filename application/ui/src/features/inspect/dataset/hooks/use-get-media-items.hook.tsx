import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';

const mediaItemsLimit = 20;

export const useGetMediaItems = () => {
    const { projectId } = useProjectIdentifier();

    const { data, isLoading, fetchNextPage, hasNextPage, isFetchingNextPage } = $api.useInfiniteQuery(
        'get',
        '/api/projects/{project_id}/images',
        {
            params: {
                query: { offset: 0, limit: mediaItemsLimit },
                path: { project_id: projectId },
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

                return pagination.offset + mediaItemsLimit;
            },
            initialPageParam: 0,
        }
    );

    const mediaItems = data?.pages.flatMap((page) => page.media) ?? [];

    return { mediaItems, isLoading, fetchNextPage, hasNextPage, isFetchingNextPage };
};
