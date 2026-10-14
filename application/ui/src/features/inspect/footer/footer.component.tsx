// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useEffect } from 'react';

import { usePatchPipeline, usePipeline, useProjectIdentifier } from '@anomalib-studio/hooks';
import { Loading, View } from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';

import { useCompletedModels } from '../../../hooks/use-completed-models.hook';
import { ConnectionStatusAdapter } from './status-bar/adapters/connection-status.adapter';
import { TrainingStatusAdapter } from './status-bar/adapters/training-status.adapter';
import { StatusBar } from './status-bar/status-bar.component';

const useDefaultModel = () => {
    const models = useCompletedModels();
    const { data: pipeline } = usePipeline();
    const { projectId } = useProjectIdentifier();
    const patchPipeline = usePatchPipeline(projectId);

    const hasSelectedModel = pipeline?.model?.id !== undefined;
    const hasNonAvailableModels = isEmpty(models.filter(({ status }) => status === 'Completed'));

    useEffect(() => {
        if (hasSelectedModel || hasNonAvailableModels || patchPipeline.isPending) {
            return;
        }

        patchPipeline.mutate({
            params: { path: { project_id: projectId } },
            body: { model_id: models[0].id },
        });
    }, [hasNonAvailableModels, hasSelectedModel, models, patchPipeline, projectId]);
};

export const Footer = () => {
    useDefaultModel();

    return (
        <View gridArea={'footer'} backgroundColor={'gray-100'} width={'100%'} height={'size-400'} overflow={'hidden'}>
            <Suspense fallback={<Loading mode={'inline'} size='S' />}>
                <ConnectionStatusAdapter />
                <TrainingStatusAdapter />
                <StatusBar />
            </Suspense>
        </View>
    );
};
