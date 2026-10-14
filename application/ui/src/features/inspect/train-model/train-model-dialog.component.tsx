import { Suspense, useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import {
    Button,
    ButtonGroup,
    Content,
    ContextualHelp,
    Dialog,
    Divider,
    Flex,
    Heading,
    Loading,
    RadioGroup,
    Text,
    View,
} from '@geti-ui/ui';
import { useSearchParams } from 'react-router';
import { toast as sonnerToast } from 'sonner';

import { TrainableModelListBox } from './trainable-model-list-box/trainable-model-list-box.component';
import { TrainingDevicePicker, useTrainingDevice } from './training-device-picker.component';
import { getDeviceKey } from './utils/device-metadata';

import classes from './train-model.module.scss';

export const TrainModelDialog = ({ close }: { close: () => void }) => {
    const [searchParams, setSearchParams] = useSearchParams();
    const { projectId } = useProjectIdentifier();
    const startTrainingMutation = $api.useMutation('post', '/api/jobs:train', {
        meta: { invalidates: [['get', '/api/jobs']] },
    });

    const { selectedDevice, setSelectedDevice, devices } = useTrainingDevice();
    const [selectedModel, setSelectedModel] = useState<string | null>(null);

    // Resolve selected key back to the full device object
    const selectedDeviceInfo = devices.find((d) => getDeviceKey(d) === selectedDevice);

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
