// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useDateFormatter } from '@geti-ui/ui';

import { useProjectTrainingJobs } from './use-project-trainingJobs.hook';
import { useTrainedModels } from './use-trained-models';
import { ModelData } from './utils';

export const useCompletedModels = () => {
    const { jobs = [] } = useProjectTrainingJobs();

    const dateFormatter = useDateFormatter({ dateStyle: 'medium', timeStyle: 'short' });

    const models = useTrainedModels()
        .filter((model) => model.is_ready)
        .map((model): ModelData | null => {
            const job = jobs.find(({ id }) => id === model.train_job_id);
            if (job === undefined) {
                return null;
            }

            let timestamp = '';
            let durationInSeconds = 0;
            const start = job.start_time ? new Date(job.start_time) : new Date();
            if (job) {
                const end = job.end_time ? new Date(job.end_time) : new Date();
                durationInSeconds = Math.floor((end.getTime() - start.getTime()) / 1000);
                timestamp = dateFormatter.format(start);
            }

            return {
                id: model.id!,
                name: model.name!,
                status: 'Completed',
                architecture: model.name!,
                startTime: start.getTime(),
                timestamp,
                durationInSeconds,
                progress: 1.0,
                job,
                sizeBytes: model.size ?? null,
                backbone: model.backbone ?? null,
            };
        })
        .filter((model): model is ModelData => model !== null);

    return models;
};
