// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api } from '@anomalib-studio/api';
import { usePipeline } from '@anomalib-studio/hooks';
import { dimensionValue, View } from '@geti-ui/ui';
import { isEmpty, isNil } from 'lodash-es';

interface FpsProp {
    projectId: string;
}

export const Fps = ({ projectId }: FpsProp) => {
    const { data: pipeline } = usePipeline();
    const isRunning = pipeline?.status === 'running';
    const formatter = new Intl.NumberFormat('en-US', {
        maximumFractionDigits: 0,
        minimumFractionDigits: 0,
    });

    const { data: metrics } = $api.useQuery(
        'get',
        '/api/projects/{project_id}/pipeline/metrics',
        { params: { path: { project_id: projectId } } },
        { enabled: isRunning, refetchInterval: 2_000 }
    );

    const requestsPerSecond = metrics?.inference.latency.latest_ms
        ? 1000 / metrics.inference.latency.latest_ms
        : undefined;

    if (isEmpty(metrics) || isNil(requestsPerSecond)) {
        return null;
    }

    return (
        <View
            top={'size-200'}
            right={'size-200'}
            zIndex={1}
            position={'absolute'}
            backgroundColor={'gray-100'}
            UNSAFE_style={{
                fontSize: dimensionValue('size-130'),
                padding: `${dimensionValue('size-85')} ${dimensionValue('size-65')}`,
                borderRadius: dimensionValue('size-25'),
            }}
        >
            {formatter.format(requestsPerSecond)} fps
        </View>
    );
};
