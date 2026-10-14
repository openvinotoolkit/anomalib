import { useMemo } from 'react';

import { $api } from '@anomalib-studio/api';
import { usePipeline } from '@anomalib-studio/hooks';
import {
    Cell,
    Column,
    Content,
    Flex,
    Heading,
    IllustratedMessage,
    Link,
    Row,
    TableBody,
    TableHeader,
    TableView,
    Text,
    View,
} from '@geti-ui/ui';
import { NotFound } from '@geti-ui/ui/icons';
import { sortBy } from 'lodash-es';
import { useDateFormatter } from 'react-aria';

import { useCompletedModels } from '../../../hooks/use-completed-models.hook';
import { useProjectTrainingJobs } from '../../../hooks/use-project-trainingJobs.hook';
import type { ModelData } from '../../../hooks/utils';
import { useRefreshModelsOnJobUpdates } from '../dataset/dataset-status-panel.component';
import { formatSize } from '../utils';
import { ModelActionsMenu } from './model-actions-menu.component';
import { ModelStatusBadges } from './model-status-badges.component';

import classes from './models-view.module.scss';

interface ModelsViewProps {
    onModelSelect: (modelId: string) => void;
}

const useShowModels = (models: ModelData[], nonCompletedJobs: ModelData[]) => {
    const { data: trainableModelsData } = $api.useSuspenseQuery('get', '/api/trainable-models');
    return sortBy(
        [...nonCompletedJobs, ...models].map((model) => {
            return {
                ...model,
                name: trainableModelsData?.trainable_models?.find((m) => m.id === model.name)?.name ?? model.name,
            };
        }),
        (model) => -model.startTime
    );
};

export const ModelsView = ({ onModelSelect }: ModelsViewProps) => {
    const { data: pipeline } = usePipeline();
    const { jobs = [] } = useProjectTrainingJobs();

    const dateFormatter = useDateFormatter({ dateStyle: 'medium', timeStyle: 'short' });
    const selectedModelId = pipeline.model?.id;
    const models = useCompletedModels();

    useRefreshModelsOnJobUpdates(jobs);

    const completedModelsJobsIDs = new Set(models.map((model) => model.job?.id));

    const nonCompletedJobs = jobs
        .filter((job) => !completedModelsJobsIDs.has(job.id))
        .map((job): ModelData => {
            const name = String(job.payload['model_name']);

            const start = job.start_time ? new Date(job.start_time) : new Date();
            const timestamp = dateFormatter.format(start);
            return {
                id: job.id!,
                name,
                status: job.status === 'pending' ? 'Training' : job.status === 'running' ? 'Training' : 'Failed',
                architecture: name,
                timestamp,
                startTime: start.getTime(),
                progress: job.progress ?? 0,
                durationInSeconds: null,
                backbone: null,
                job,
                sizeBytes: null,
            };
        });

    const showModels = useShowModels(models, nonCompletedJobs);

    const tableSelectedKeys = useMemo(() => {
        if (selectedModelId === undefined) {
            return new Set<string>();
        }

        return new Set<string>([selectedModelId]);
    }, [selectedModelId]);

    return (
        <View backgroundColor='gray-100' height='100%'>
            {/* Models Table */}
            <TableView
                aria-label='Models'
                overflowMode='wrap'
                selectionStyle='highlight'
                selectionMode='single'
                minHeight={showModels.length === 0 ? 'size-3600' : 'auto'}
                selectedKeys={tableSelectedKeys}
                UNSAFE_className={classes.table}
                renderEmptyState={() => (
                    <IllustratedMessage>
                        <NotFound />
                        <Heading>No models yet</Heading>
                        <Content>Train a model to see it here.</Content>
                    </IllustratedMessage>
                )}
            >
                <TableHeader>
                    <Column width='2fr'>MODEL NAME</Column>
                    <Column align='end' width='1fr'>
                        MODEL SIZE
                    </Column>
                    <Column aria-label='Model actions' width='0fr'>
                        {' '}
                    </Column>
                </TableHeader>
                <TableBody items={showModels}>
                    {(model) => (
                        <Row key={model.id}>
                            <Cell>
                                <Flex alignItems='start' gap='size-50' direction='column' minWidth={0}>
                                    <Flex alignItems='end' gap='size-75' minWidth={0} width='100%'>
                                        {model.status === 'Completed' ? (
                                            <Link
                                                variant='secondary'
                                                onPress={() => onModelSelect(model.id)}
                                                isQuiet
                                                UNSAFE_className={classes.modelName}
                                            >
                                                {model.name}
                                            </Link>
                                        ) : (
                                            <Text marginTop={'size-25'} UNSAFE_className={classes.modelName}>
                                                {model.name}
                                            </Text>
                                        )}
                                        <ModelStatusBadges
                                            isSelected={selectedModelId === model.id}
                                            jobStatus={model.job?.status}
                                            progress={model.progress}
                                        />
                                    </Flex>
                                    <Text UNSAFE_className={classes.modelTimestamp}>{model.timestamp}</Text>
                                </Flex>
                            </Cell>
                            <Cell>
                                <Text UNSAFE_className={classes.modelSize}>{formatSize(model.sizeBytes)}</Text>
                            </Cell>
                            <Cell>
                                <Flex justifyContent='end' alignItems='center'>
                                    <Flex alignItems='center' gap='size-200'>
                                        <ModelActionsMenu model={model} selectedModelId={selectedModelId} />
                                    </Flex>
                                </Flex>
                            </Cell>
                        </Row>
                    )}
                </TableBody>
            </TableView>
        </View>
    );
};
