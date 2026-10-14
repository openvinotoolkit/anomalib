// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { Flex, Heading, Loading, View } from '@geti-ui/ui';

import { TrainModelButton } from '../train-model/train-model-button/train-model-button.component';
import { DatasetList } from './dataset-list.component';
import { UploadImages } from './upload-images.component';

export const Dataset = () => {
    return (
        <Flex direction={'column'} height={'100%'}>
            <Heading margin={0}>
                <Flex justifyContent={'space-between'}>
                    Dataset
                    <Flex gap='size-200'>
                        <UploadImages />
                        <TrainModelButton />
                    </Flex>
                </Flex>
            </Heading>
            <Suspense fallback={<Loading mode={'inline'} />}>
                <View flex={1} padding={'size-300'}>
                    <Flex direction={'column'} height={'100%'} gap={'size-300'}>
                        <DatasetList />
                    </Flex>
                </View>
            </Suspense>
        </Flex>
    );
};
