// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { usePipeline, useProjectIdentifier } from '@anomalib-studio/hooks';
import { dimensionValue, Divider, Flex, View } from '@geti-ui/ui';
import { isNil } from 'lodash-es';
import { useIsPipelineConfigured } from 'src/hooks/use-is-pipeline-configured.hook';

import { useActivePipelineStatus } from '../main-content/hooks/use-active-pipeline-status.hook';
import { AnomalyMap } from './anomaly-map/anomaly-map.component';
import { InferenceDevices } from './inference-devices/inference-devices.component';
import { PipelineConfiguration } from './pipeline-configuration.component';
import { TogglePipelineButton } from './toggle-pipeline-button/toggle-pipeline-button.component';

export const Toolbar = () => {
    const { projectId } = useProjectIdentifier();
    const { data: pipeline } = usePipeline();
    const canEnablePipeline = useIsPipelineConfigured(pipeline);
    const { hasActiveProject, activeProjectId } = useActivePipelineStatus(projectId);

    const hasModel = !isNil(pipeline?.model?.id);
    const hideTogglePipeline = hasActiveProject && activeProjectId !== projectId;

    return (
        <View
            gridArea='toolbar'
            padding='size-200'
            backgroundColor={'gray-100'}
            UNSAFE_style={{ fontSize: dimensionValue('size-150'), color: 'var(--spectrum-global-color-gray-800)' }}
        >
            <Flex height='100%' gap='size-200' alignItems={'center'} justifyContent={'space-between'}>
                <Flex gap={'size-200'}>
                    {hasModel && (
                        <>
                            <InferenceDevices />
                            <Divider size={'S'} orientation={'vertical'} />
                            <AnomalyMap />
                        </>
                    )}
                </Flex>

                <Flex gap={'size-200'}>
                    {!hideTogglePipeline && canEnablePipeline && <TogglePipelineButton />}
                    <PipelineConfiguration />
                </Flex>
            </Flex>
        </View>
    );
};
