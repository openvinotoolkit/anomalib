import { Suspense, useState } from 'react';

import { Flex, Heading, Loading } from '@geti-ui/ui';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import { useCompletedModels } from '../../../hooks/use-completed-models.hook';
import { TrainModelButton } from '../train-model/train-model-button/train-model-button.component';
import { ModelDetail } from './model-detail/model-detail.component';
import { ModelsView } from './models-view.component';

export const Models = () => {
    const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
    const models = useCompletedModels();
    const { data: pipeline } = usePipeline();

    const selectedModel = models.find((m) => m.id === selectedModelId);
    const isActiveModel = pipeline.model?.id === selectedModelId;

    const handleModelSelect = (modelId: string) => {
        setSelectedModelId(modelId);
    };

    const handleBack = () => {
        setSelectedModelId(null);
    };

    if (selectedModel) {
        return (
            <Suspense
                fallback={
                    <Loading
                        mode={'overlay'}
                        size='M'
                        style={{ backgroundColor: 'var(--spectrum-global-color-gray-100)' }}
                    />
                }
            >
                <ModelDetail model={selectedModel} isActiveModel={isActiveModel} onBack={handleBack} />
            </Suspense>
        );
    }

    return (
        <Flex direction={'column'} height={'100%'} gap={'size-250'}>
            <Heading margin={0}>
                <Flex justifyContent={'space-between'}>
                    Models
                    <Flex gap='size-200'>
                        <TrainModelButton />
                    </Flex>
                </Flex>
            </Heading>
            <Suspense
                fallback={
                    <Loading
                        mode={'overlay'}
                        size='M'
                        style={{ backgroundColor: 'var(--spectrum-global-color-gray-100)' }}
                    />
                }
            >
                <Flex direction={'column'} height={'100%'} gap={'size-300'}>
                    <ModelsView onModelSelect={handleModelSelect} />
                </Flex>
            </Suspense>
        </Flex>
    );
};
