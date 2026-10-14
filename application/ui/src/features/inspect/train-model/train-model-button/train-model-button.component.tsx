// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense } from 'react';

import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { Button, Content, ContextualHelp, DialogTrigger, Flex, Heading, Text } from '@geti-ui/ui';

import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from '../../dataset/utils';
import { pluralize } from '../../utils';
import { TrainModelDialog } from '../train-model-dialog.component';

const useTrainingButtonStatus = () => {
    const { projectId } = useProjectIdentifier();
    const { data } = $api.useQuery('get', '/api/projects/{project_id}/images', {
        params: { path: { project_id: projectId }, query: { limit: 1, offset: 0 } },
    });

    const uploadedNormalImages = data?.pagination.total ?? data?.media.length ?? 0;
    const missingNormalImages = Math.max(
        0,
        REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING - uploadedNormalImages
    );

    return {
        isDisabled: missingNormalImages > 0,
        missingNormalImages,
    };
};

export const TrainModelButton = () => {
    const { isDisabled, missingNormalImages } = useTrainingButtonStatus();

    // eslint-disable-next-line max-len
    const message = `Add ${missingNormalImages} more normal ${pluralize(missingNormalImages, 'image', 'images')} to start training.`;

    return (
        <Suspense
            fallback={
                <Button isDisabled isPending>
                    Train model
                </Button>
            }
        >
            <Flex gap='size-100' alignItems='center'>
                <DialogTrigger type='modal'>
                    <Button isDisabled={isDisabled}>Train model</Button>
                    {(close) => <TrainModelDialog close={close} />}
                </DialogTrigger>

                {isDisabled && (
                    <ContextualHelp variant='info'>
                        <Heading>More images required</Heading>
                        <Content>
                            <Text>{message}</Text>
                        </Content>
                    </ContextualHelp>
                )}
            </Flex>
        </Suspense>
    );
};
