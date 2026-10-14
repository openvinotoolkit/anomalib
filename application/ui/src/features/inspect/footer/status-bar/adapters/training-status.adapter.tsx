// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect } from 'react';

import { $api } from '@anomalib-studio/api';
import { SchemaJob as Job } from '@anomalib-studio/api/spec';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { queryOptions, experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';
import { fetchSSE } from 'src/api/fetch-sse';

import { useStatusBar } from '../status-bar-context';

interface JobProgress {
    progress: number;
    message: string;
}

export const TrainingStatusAdapter = () => {
    const { setStatus, removeStatus } = useStatusBar();
    const { projectId } = useProjectIdentifier();

    const { data: jobsData } = $api.useQuery('get', '/api/jobs', undefined, {
        refetchInterval: 10000,
    });

    const { mutate: cancelJob } = $api.useMutation('post', '/api/jobs/{job_id}:cancel', {
        meta: { invalidates: [['get', '/api/jobs']] },
    });

    const trainingJob = jobsData?.jobs.find(
        (job: Job) =>
            job.project_id === projectId &&
            job.type === 'training' &&
            (job.status === 'running' || job.status === 'pending')
    );

    const { data: progressData } = useQuery(
        queryOptions({
            queryKey: ['get', '/api/jobs/{job_id}/progress', trainingJob?.id],
            queryFn: streamedQuery<JobProgress>({
                streamFn: () => fetchSSE<JobProgress>(`/api/jobs/${trainingJob?.id}/progress`),
            }),
            enabled: !!trainingJob?.id,
            staleTime: Infinity,
        })
    );

    const latestProgress = progressData?.at(-1);
    const progress = latestProgress?.progress ?? trainingJob?.progress;
    const message = latestProgress?.message ?? trainingJob?.message;

    useEffect(() => {
        if (!trainingJob) {
            removeStatus('training');
            return;
        }

        const jobId = trainingJob.id;
        if (!jobId) {
            return;
        }

        setStatus({
            id: 'training',
            type: 'training',
            message: `Training ${trainingJob.payload.model_name}...`,
            detail: message,
            progress,
            variant: 'info',
            isCancellable: true,
            onCancel: () => {
                if (trainingJob.id) {
                    cancelJob({ params: { path: { job_id: trainingJob.id } } });
                }
            },
        });
    }, [
        trainingJob?.id,
        trainingJob?.payload.model_name,
        message,
        progress,
        trainingJob?.status,
        setStatus,
        removeStatus,
        cancelJob,
    ]);

    return null;
};
