// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { Button, Flex, Grid, minmax, repeat } from '@geti-ui/ui';
import { ChevronDownLight, ChevronUpLight } from '@geti-ui/ui/icons';

import { TrainableModelCard } from '../trainable-model-card.component';

interface TrainableModelListBoxProps {
    selectedModelTemplateId: string | null;
}

export const TrainableModelListBox = ({ selectedModelTemplateId }: TrainableModelListBoxProps) => {
    const [showMore, setShowMore] = useState(false);

    const { data } = $api.useSuspenseQuery('get', '/api/trainable-models');

    const models = data.trainable_models;
    const recommendedModels = models.filter((m) => m.recommended);
    const otherModels = models.filter((m) => !m.recommended);

    const displayedModels = showMore ? models : recommendedModels;

    return (
        <Flex direction='column' gap='size-200'>
            <Grid columns={repeat('auto-fit', minmax('size-3600', '1fr'))} gap='size-200'>
                {displayedModels.map((model) => (
                    <TrainableModelCard
                        key={model.id}
                        model={model}
                        isSelected={selectedModelTemplateId === model.id}
                    />
                ))}
            </Grid>

            {otherModels.length > 0 && (
                <Button variant='secondary' onPress={() => setShowMore(!showMore)} alignSelf={'center'}>
                    {showMore ? (
                        <Flex alignItems='center' gap='size-75'>
                            <ChevronUpLight /> Show less
                        </Flex>
                    ) : (
                        <Flex alignItems='center' gap='size-75'>
                            <ChevronDownLight /> Show more ({otherModels.length} models)
                        </Flex>
                    )}
                </Button>
            )}
        </Flex>
    );
};
