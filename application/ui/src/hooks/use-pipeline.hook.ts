// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { toast } from '@anomalib-studio/toast';
import { isString } from 'lodash-es';

import { $api } from '../api/client';
import { useProjectIdentifier } from './use-project-identifier.hook';

export const usePipeline = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useSuspenseQuery('get', '/api/projects/{project_id}/pipeline', {
        params: { path: { project_id: projectId } },
    });
};

const POLLING_INTERVAL = 5000;
export const usePipelineMetrics = () => {
    const { projectId } = useProjectIdentifier();

    return $api.useQuery(
        'get',
        '/api/projects/{project_id}/pipeline/metrics',
        {
            params: { path: { project_id: projectId } },
        },
        {
            refetchInterval: (query) => (query.state.status === 'success' ? POLLING_INTERVAL : false),
            retry: false,
        }
    );
};

export const usePatchPipeline = (project_id: string) => {
    return $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        onError: (error) => {
            if (error) {
                toast({ type: 'error', message: String(error.detail) });
            }
        },
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id } } }]],
        },
    });
};

export const useRunPipeline = ({ onSuccess }: { onSuccess?: () => void }) => {
    const { projectId } = useProjectIdentifier();

    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:run', {
        onSuccess,
        onError: (error) => {
            if (error) {
                toast({ type: 'error', message: String(error.detail) });
            }
        },
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
                ['get', '/api/active-pipeline'],
            ],
        },
    });
};

export const useActivatePipeline = ({ onSuccess }: { onSuccess?: () => void }) => {
    const { projectId } = useProjectIdentifier();
    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:activate', {
        onSuccess,
        onError: (error) => {
            if (error) {
                const message = isString(error.detail) ? String(error.detail) : String(error);
                toast({ type: 'error', message });
            }
        },
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
                ['get', '/api/active-pipeline'],
            ],
        },
    });
};

export const useActivateAndRunPipeline = ({ onSuccess }: { onSuccess?: () => void } = {}) => {
    const { projectId } = useProjectIdentifier();
    const activatePipeline = useActivatePipeline({});
    const runPipeline = useRunPipeline({ onSuccess });

    const mutateAsync = async () => {
        const params = { params: { path: { project_id: projectId } } };

        await activatePipeline.mutateAsync(params);
        return runPipeline.mutateAsync(params);
    };

    return {
        mutateAsync,
        isPending: activatePipeline.isPending || runPipeline.isPending,
        isError: activatePipeline.isError || runPipeline.isError,
        error: activatePipeline.error ?? runPipeline.error,
        activatePipeline,
        runPipeline,
    };
};

export const useDisablePipeline = (project_id: string) => {
    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:disable', {
        onError: (error) => {
            if (error) {
                const message = isString(error.detail) ? String(error.detail) : String(error);
                toast({ type: 'error', message });
            }
        },
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id } } }],
                ['get', '/api/active-pipeline'],
            ],
        },
    });
};

export const useStopPipeline = (project_id: string) => {
    return $api.useMutation('post', '/api/projects/{project_id}/pipeline:stop', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id } } }]],
        },
    });
};

export const useConnectSourceToPipeline = () => {
    const { projectId } = useProjectIdentifier();
    const pipeline = usePatchPipeline(projectId);

    return (source_id: string) =>
        pipeline.mutateAsync({ params: { path: { project_id: projectId } }, body: { source_id } });
};

export const useConnectSinkToPipeline = () => {
    const { projectId } = useProjectIdentifier();
    const pipeline = usePatchPipeline(projectId);

    return (sink_id: string) =>
        pipeline.mutateAsync({ params: { path: { project_id: projectId } }, body: { sink_id } });
};

export const useActivePipeline = () => {
    return $api.useSuspenseQuery('get', '/api/active-pipeline');
};
