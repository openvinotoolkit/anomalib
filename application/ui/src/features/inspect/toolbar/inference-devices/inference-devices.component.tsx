import { useEffect, useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import { Content, ContextualHelp, Heading, Item, Key, Picker, Text } from '@geti-ui/ui';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import { getDeviceLabel } from '../../train-model/utils/device-metadata';

/**
 * Find the openvino_name that matches the pipeline's inference_device value.
 * The backend stores inference_device in uppercase (e.g. "CPU", "GPU.0"),
 * and openvino_name uses the same convention, so we compare case-insensitively.
 */
const findMatchingKey = (
    pipelineDevice: string | null | undefined,
    options: { id: string | null | undefined }[]
): string | null => {
    if (!pipelineDevice) return null;
    const normalized = pipelineDevice.toUpperCase();
    const match = options.find((o) => o.id?.toUpperCase() === normalized);
    return match?.id ?? null;
};

export const InferenceDevices = () => {
    const { data: inferenceDevices } = $api.useSuspenseQuery('get', '/api/system/devices/inference');
    const { data: trainingDevices } = $api.useSuspenseQuery('get', '/api/system/devices/training');
    const { data: pipeline } = usePipeline();
    const { projectId } = useProjectIdentifier();

    // Optimistic key override while mutation is in-flight
    const [optimisticKey, setOptimisticKey] = useState<Key | null>(null);

    const options = inferenceDevices.map((device) => {
        const id = device.openvino_name;
        const label = getDeviceLabel(device);
        return { id, label };
    });

    // Check if NVIDIA GPUs are available for training but absent from inference
    const hasNvidiaTrainingDevices = trainingDevices.some((d) => d.type === 'cuda');

    // Derive selected key from pipeline config; use optimistic override during mutation
    const derivedKey = findMatchingKey(pipeline.inference_device, options);
    const selectedKey = optimisticKey ?? derivedKey;

    // Clear optimistic override only once pipeline data reflects it (avoids flicker before refetch)
    useEffect(() => {
        if (optimisticKey !== null && derivedKey === optimisticKey) {
            setOptimisticKey(null);
        }
    }, [optimisticKey, derivedKey]);

    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            ],
        },
        onError: (error) => {
            setOptimisticKey(null);
            if (error) {
                toast({ type: 'error', message: String(error.detail) });
            }
        },
    });

    const handleChange = (key: Key | null) => {
        if (key === null) {
            return;
        }

        setOptimisticKey(key);
        updatePipeline.mutate({
            params: { path: { project_id: projectId } },
            body: { inference_device: key },
        });
    };

    return (
        <Picker
            width='auto'
            minWidth='size-2400'
            label='Inference device: '
            aria-label='inference devices'
            labelAlign='end'
            labelPosition='side'
            items={options}
            onSelectionChange={handleChange}
            selectedKey={selectedKey}
            contextualHelp={
                hasNvidiaTrainingDevices ? (
                    <ContextualHelp variant='info'>
                        <Heading>NVIDIA GPUs</Heading>
                        <Content>
                            <Text>
                                NVIDIA GPUs are available for training but are not supported for inference. Inference
                                uses OpenVINO, which supports CPU and Intel GPU/NPU devices.
                            </Text>
                        </Content>
                    </ContextualHelp>
                ) : undefined
            }
        >
            {(item) => <Item>{item.label}</Item>}
        </Picker>
    );
};
