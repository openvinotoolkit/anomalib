import { ComponentProps, Suspense, useEffect, useRef } from 'react';

import { SchemaJob as Job, SchemaJob, SchemaJobStatus } from '@anomalib-studio/api/spec';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { Content, Flex, Heading, InlineAlert, IntelBrandedLoading, ProgressBar, Text } from '@geti-ui/ui';
import { useQueryClient } from '@tanstack/react-query';
import { isEqual } from 'lodash-es';

import { useProjectTrainingJobs } from '../../../hooks/use-project-trainingJobs.hook';
import { ShowJobLogs } from '../jobs/show-job-logs.component';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

interface NotEnoughNormalImagesToTrainProps {
    mediaItemsCount: number;
}

const NotEnoughNormalImagesToTrain = ({ mediaItemsCount }: NotEnoughNormalImagesToTrainProps) => {
    const missingNormalImages = REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING - mediaItemsCount;

    return (
        <InlineAlert variant='neutral'>
            <Heading>{missingNormalImages} images required</Heading>
            <Content>
                Capture {missingNormalImages} images of normal cases. They help the model learn what is standard, so it
                can better detect anomalies.
            </Content>
        </InlineAlert>
    );
};

interface TrainingInProgressProps {
    job: Job;
}

const statusToVariant: Record<SchemaJobStatus, ComponentProps<typeof InlineAlert>['variant']> = {
    pending: 'info',
    running: 'info',
    completed: 'positive',
    canceled: 'negative',
    failed: 'negative',
};

function getHeading(job: SchemaJob) {
    if (job.status === 'pending') {
        return `Training will start soon - ${job.payload.model_name}`;
    }
    if (job.status === 'running') {
        return `Training in progress - ${job.payload.model_name}`;
    }

    if (job.status === 'failed') {
        return `Training failed - ${job.payload.model_name}`;
    }

    if (job.status === 'canceled') {
        return `Training canceled - ${job.payload.model_name}`;
    }

    if (job.status === 'completed') {
        return `Training completed - ${job.payload.model_name}`;
    }
    return null;
}

const TrainingInProgress = ({ job }: TrainingInProgressProps) => {
    if (job === undefined) {
        return null;
    }

    const variant = statusToVariant[job.status];
    const heading = getHeading(job);

    return (
        <InlineAlert variant={variant}>
            <Heading>
                <Flex gap='size-100' alignItems={'center'} justifyContent={'space-between'}>
                    {heading}
                    {job.id && <ShowJobLogs jobId={job.id} />}
                </Flex>
            </Heading>
            <Content>
                <Flex direction={'column'} gap={'size-100'}>
                    <Text>{job.message}</Text>
                    {job.status === 'pending' && <ProgressBar aria-label='Training progress' isIndeterminate />}
                </Flex>
            </Content>
        </InlineAlert>
    );
};

export const useRefreshModelsOnJobUpdates = (jobs: Job[] | undefined) => {
    const queryClient = useQueryClient();
    const { projectId } = useProjectIdentifier();
    const prevJobsRef = useRef<Job[]>([]);

    useEffect(() => {
        if (jobs === undefined) {
            return;
        }

        if (!isEqual(prevJobsRef.current, jobs)) {
            const shouldRefetchModels = jobs.some((job, idx) => {
                // NOTE: assuming index stays the same
                return job.status === 'completed' && job.status !== prevJobsRef.current.at(idx)?.status;
            });

            if (shouldRefetchModels) {
                queryClient.invalidateQueries({
                    queryKey: [
                        'get',
                        '/api/projects/{project_id}/models',
                        { params: { path: { project_id: projectId } } },
                    ],
                });
            }
        }

        prevJobsRef.current = jobs ?? [];
    }, [jobs, queryClient, projectId]);
};

const TrainingInProgressList = () => {
    const { jobs } = useProjectTrainingJobs();
    useRefreshModelsOnJobUpdates(jobs);

    if (jobs === undefined || jobs.length === 0) {
        return null;
    }

    return (
        <Flex direction={'column'} gap={'size-50'} UNSAFE_style={{ overflowY: 'auto' }}>
            {jobs?.map((job) => (
                <TrainingInProgress job={job} key={job.id} />
            ))}
        </Flex>
    );
};

interface DatasetStatusPanelProps {
    mediaItemsCount: number;
}

export const DatasetStatusPanel = ({ mediaItemsCount }: DatasetStatusPanelProps) => {
    if (mediaItemsCount < REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING) {
        return <NotEnoughNormalImagesToTrain mediaItemsCount={mediaItemsCount} />;
    }

    return (
        <Suspense fallback={<IntelBrandedLoading />}>
            <TrainingInProgressList />
        </Suspense>
    );
};
