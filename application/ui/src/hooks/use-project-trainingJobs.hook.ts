// Copyright (C) 2020-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useQuery } from '@tanstack/react-query';

import { fetchClient } from '../api/client';
import { SchemaJob } from '../api/openapi-spec';
import { useProjectIdentifier } from './use-project-identifier.hook';

const REFETCH_INTERVAL_WITH_TRAINING = 5_000;
const JOBS_PAGE_LIMIT = 50;

const getAllJobs = async (signal: AbortSignal): Promise<SchemaJob[]> => {
    let offset = 0;
    const jobs: SchemaJob[] = [];
    const seenIds = new Set<string>();

    while (true) {
        signal.throwIfAborted();
        const { data, error } = await fetchClient.GET('/api/jobs', {
            params: { query: { offset, limit: JOBS_PAGE_LIMIT } },
            signal,
        });

        if (error !== undefined) {
            throw error;
        }

        const { pagination } = data;
        const nextOffset = pagination.offset + pagination.count;

        if (pagination.offset !== offset || (nextOffset < pagination.total && nextOffset <= offset)) {
            throw new Error('Unable to retrieve all jobs: pagination did not advance as expected.');
        }

        for (const job of data.jobs) {
            if (job.id == null || !seenIds.has(job.id)) {
                jobs.push(job);
                if (job.id != null) {
                    seenIds.add(job.id);
                }
            }
        }

        if (nextOffset >= pagination.total) {
            return jobs;
        }
        offset = nextOffset;
    }
};

/** Fetch every job page before exposing project jobs; offset pagination is not a consistent snapshot. */
export const useProjectTrainingJobs = (options: { refetchInterval?: number; refetchOnMount?: 'always' } = {}) => {
    const { projectId } = useProjectIdentifier();

    const { data, ...query } = useQuery({
        // Isolate the aggregate from raw pages, while retaining existing job-mutation invalidation.
        queryKey: ['get', '/api/jobs', 'all'],
        queryFn: ({ signal }) => getAllJobs(signal),
        refetchOnMount: options.refetchOnMount,
        refetchInterval: ({ state }) => {
            const projectHasTrainingJob = state.data?.some(
                ({ project_id, type, status }) =>
                    projectId === project_id && type === 'training' && (status === 'running' || status === 'pending')
            );

            return options.refetchInterval ?? (projectHasTrainingJob ? REFETCH_INTERVAL_WITH_TRAINING : undefined);
        },
    });

    return { ...query, jobs: data?.filter((job) => job.project_id === projectId) };
};
