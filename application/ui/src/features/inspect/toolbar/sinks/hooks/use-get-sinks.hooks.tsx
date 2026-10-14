import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';

import { SinkConfig } from '../utils';

const sinksItemsLimit = 20;

export const useGetSinks = () => {
    const { projectId } = useProjectIdentifier();

    const { data, isLoading, fetchNextPage, hasNextPage, isFetchingNextPage } = $api.useInfiniteQuery(
        'get',
        '/api/projects/{project_id}/sinks',
        {
            params: {
                query: { offset: 0, limit: sinksItemsLimit },
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

                return pagination.offset + sinksItemsLimit;
            },
            initialPageParam: 0,
        }
    );

    // Filter out ROS sinks as they're not supported yet
    const sinks = (data?.pages.flatMap((page) => page.sinks) ?? []).filter(
        (sink): sink is SinkConfig => sink.sink_type !== 'ros'
    );

    return { sinks, isLoading, fetchNextPage, hasNextPage, isFetchingNextPage };
};
