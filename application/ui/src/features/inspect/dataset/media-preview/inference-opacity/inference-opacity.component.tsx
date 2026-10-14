// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, DialogTrigger, Flex, Slider, Text, View } from '@geti-ui/ui';
import { ChevronDownSmall } from '@geti-ui/ui/icons';

import { useInference } from '../providers/inference-opacity-provider.component';

interface InferenceOpacityProps {
    isDisabled: boolean;
}

export const InferenceOpacity = ({ isDisabled }: InferenceOpacityProps) => {
    const { inferenceOpacity, onInferenceOpacityChange } = useInference();

    return (
        <Flex alignItems={'center'} gap={'size-50'}>
            <Text>Opacity:</Text>
            <DialogTrigger type={'popover'} placement={'bottom'}>
                <ActionButton width={'size-800'} isDisabled={isDisabled}>
                    <Flex alignItems={'center'} gap={'size-50'}>
                        <span>{Math.floor(inferenceOpacity * 100)}%</span>
                        <Flex>
                            <ChevronDownSmall style={{ order: 1 }} />
                        </Flex>
                    </Flex>
                </ActionButton>
                <View padding={'size-100'}>
                    <Slider
                        value={inferenceOpacity}
                        onChange={onInferenceOpacityChange}
                        maxValue={1}
                        minValue={0}
                        step={0.01}
                    />
                </View>
            </DialogTrigger>
        </Flex>
    );
};
