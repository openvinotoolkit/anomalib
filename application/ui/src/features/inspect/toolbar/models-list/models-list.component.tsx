// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePatchPipeline, usePipeline, useProjectIdentifier } from '@anomalib-studio/hooks';
import { Button, Content, IllustratedMessage } from '@geti-ui/ui';
import { NotFound } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { isEmpty } from 'lodash-es';

import { useGetModels } from '../../../..//hooks/use-get-models.hook';
import { LoadMoreList } from '../../../../components/load-more-list/load-more-list.component';

import classes from './model-list.module.scss';

export const ModelsList = () => {
    const { models, isLoading, hasNextPage, fetchNextPage } = useGetModels();
    const { projectId } = useProjectIdentifier();
    const patchPipeline = usePatchPipeline(projectId);
    const { data: pipeline } = usePipeline();

    const selectedModelId = pipeline.model?.id;
    const modelsIds = models.map((model) => model.id).filter(Boolean) as string[];

    const handleSelectionChange = (model_id: string) => {
        patchPipeline.mutate({
            params: { path: { project_id: projectId } },
            body: { model_id },
        });
    };

    if (isEmpty(modelsIds)) {
        return <EmptyState />;
    }

    return (
        <LoadMoreList isLoading={isLoading} hasNextPage={hasNextPage} onLoadMore={fetchNextPage}>
            {models.map((model) => (
                <Button
                    key={model.id}
                    variant='secondary'
                    onPress={() => handleSelectionChange(String(model.id))}
                    height={'size-800'}
                    isPending={patchPipeline.isPending}
                    isDisabled={patchPipeline.isPending}
                    UNSAFE_className={clsx(classes.option, { [classes.activeCard]: model.id === selectedModelId })}
                >
                    {model.name}
                </Button>
            ))}
        </LoadMoreList>
    );
};

const EmptyState = () => {
    return (
        <IllustratedMessage>
            <NotFound />
            <Content>No models trained yet</Content>
        </IllustratedMessage>
    );
};
