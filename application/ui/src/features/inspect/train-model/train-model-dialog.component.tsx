// Copyright (C) 2020-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { useActivePipeline, useProjectIdentifier } from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import {
    Button,
    ButtonGroup,
    Content,
    ContextualHelp,
    Dialog,
    Divider,
    Flex,
    Heading,
    InlineAlert,
    Loading,
    RadioGroup,
    Text,
    View,
} from '@geti-ui/ui';
import { useSearchParams } from 'react-router';
import { toast as sonnerToast } from 'sonner';
import { useProjectTrainingJobs } from 'src/hooks/use-project-trainingJobs.hook';

import { TrainableModelListBox } from './trainable-model-list-box/trainable-model-list-box.component';
import { TrainingDevicePicker, useTrainingDevice } from './training-device-picker.component';
import { getDeviceKey, isSamePhysicalDevice } from './utils/device-metadata';

import classes from './train-model.module.scss';

const useTrainJob = () => {
    return $api.useMutation('post', '/api/jobs:train', {
        meta: { invalidates: [['get', '/api/jobs']] },
        onError: (error) => {
            const message =
                'message' in error
                    ? String(error.message)
                    : String(error.detail ?? 'Unable to start training. Please try again.');

            toast({ type: 'error', message });
        },
    });
};

export const TrainModelDialog = ({ close }: { close: () => void }) => {
    const startTrainingMutation = useTrainJob();
    const { projectId } = useProjectIdentifier();
    const { jobs = [] } = useProjectTrainingJobs();
    const [searchParams, setSearchParams] = useSearchParams();
    const [selectedModel, setSelectedModel] = useState<string | null>(null);
    const { selectedDevice, setSelectedDevice, devices } = useTrainingDevice();

    // Resolve selected key back to the full device object
    const selectedDeviceInfo = devices.find((device) => getDeviceKey(device) === selectedDevice);

    // Warning 1: active training job for this project
    const hasActiveTrainingJob = jobs.some(
        (job) => job.type === 'training' && (job.status === 'running' || job.status === 'pending')
    );

    // Warning 2: pipeline running on the same device as selected for training
    const { data: pipeline } = useActivePipeline();

    const pipelineIsActive = pipeline?.status === 'running' || pipeline?.status === 'active';
    const isDeviceConflict = pipelineIsActive && isSamePhysicalDevice(selectedDeviceInfo, pipeline?.inference_device);

    const startTraining = async () => {
        if (selectedModel === null || selectedDeviceInfo === undefined) {
            return;
        }

        await startTrainingMutation.mutateAsync({
            body: {
                project_id: projectId,
                model_name: selectedModel,
                device: { type: selectedDeviceInfo.type, index: selectedDeviceInfo.index },
            },
        });

        close();
        sonnerToast.dismiss();

        searchParams.set('mode', 'Models');
        setSearchParams(searchParams);
    };

    const isStartDisabled =
        selectedModel === null || selectedDeviceInfo === undefined || startTrainingMutation.isPending;

    return (
        <Dialog size='L' UNSAFE_style={{ width: 'fit-content' }}>
            <Heading>Train model</Heading>
            <Divider />
            <Content UNSAFE_style={{ width: 'fit-content' }}>
                <View
                    padding={'size-250'}
                    backgroundColor={'gray-50'}
                    flex={1}
                    minHeight={0}
                    overflow={'hidden auto'}
                    minWidth={'60vw'}
                >
                    <Flex direction='column' gap='size-300'>
                        {hasActiveTrainingJob && (
                            <InlineAlert variant='info'>
                                <Heading level={5}>Training already queued</Heading>
                                <Text>
                                    A training job is already pending or running for this project. A different model or
                                    device can be queued, but the same training configuration cannot be submitted again
                                    until the current job finishes.
                                </Text>
                            </InlineAlert>
                        )}

                        <Flex direction='column' gap='size-150'>
                            <Flex alignItems='center' gap='size-100'>
                                <Heading level={4} margin={0}>
                                    Select model
                                </Heading>
                                <ContextualHelp variant='info'>
                                    <Heading>Recommended models</Heading>
                                    <Content>
                                        <Text>
                                            Recommended models consistently provide strong accuracy with a practical
                                            balance of training and inference efficiency.
                                        </Text>
                                    </Content>
                                </ContextualHelp>
                            </Flex>
                            <RadioGroup
                                isEmphasized
                                aria-label={`Select a model to train`}
                                onChange={(modelId) => {
                                    setSelectedModel(modelId);
                                }}
                                value={selectedModel}
                                minWidth={0}
                                width='100%'
                                UNSAFE_className={classes.radioGroup}
                            >
                                <Suspense fallback={<Loading mode='inline' />}>
                                    <TrainableModelListBox selectedModelTemplateId={selectedModel} />
                                </Suspense>
                            </RadioGroup>
                        </Flex>
                        <Flex direction='column' gap='size-150'>
                            <Heading level={4} margin={0}>
                                Training device
                            </Heading>
                            <TrainingDevicePicker
                                selectedDevice={selectedDevice}
                                onDeviceChange={setSelectedDevice}
                                devices={devices}
                            />

                            {isDeviceConflict && (
                                <InlineAlert variant='notice'>
                                    <Heading level={5}>
                                        Inference pipeline is running on {pipeline?.inference_device}
                                    </Heading>
                                    <Text>
                                        Training on the same device may reduce inference performance. Consider switching
                                        to a different device or stopping the pipeline before training.
                                    </Text>
                                </InlineAlert>
                            )}
                        </Flex>
                    </Flex>
                </View>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    onPress={startTraining}
                    // eslint-disable-next-line jsx-a11y/no-autofocus
                    autoFocus
                    isPending={startTrainingMutation.isPending}
                    isDisabled={isStartDisabled}
                >
                    Start
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
