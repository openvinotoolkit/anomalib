// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';

const modelItemsLimit = 50;

export const useGetModels = () => {
    const { projectId } = useProjectIdentifier();

    const { data, isLoading, fetchNextPage, hasNextPage, isFetchingNextPage } = $api.useInfiniteQuery(
        'get',
        '/api/projects/{project_id}/models',
        {
            params: {
                query: { offset: 0, limit: modelItemsLimit },
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

                return pagination.offset + modelItemsLimit;
            },
            initialPageParam: 0,
        }
    );

    const models = data?.pages.flatMap((page) => page.models) ?? [];

    return { models, isLoading, fetchNextPage, hasNextPage, isFetchingNextPage };
};
