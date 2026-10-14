// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { Button, Flex, Form } from '@geti-ui/ui';

import { useConnectSourceToPipeline } from '../../../../../hooks/use-pipeline.hook';
import { useSourceAction } from '../hooks/use-source-action.hook';
import { SourceConfig } from '../util';

interface AddSourceProps<T> {
    config: Awaited<T>;
    onSaved: () => void;
    componentFields: (state: Awaited<T>) => ReactNode;
    bodyFormatter: (formData: FormData) => T;
}

export const AddSource = <T extends SourceConfig>({
    config,
    onSaved,
    bodyFormatter,
    componentFields,
}: AddSourceProps<T>) => {
    const connectToPipelineMutation = useConnectSourceToPipeline();

    const [state, submitAction, isPending] = useSourceAction({
        config,
        isNewSource: true,
        onSaved: async (sourceId) => {
            await connectToPipelineMutation(sourceId);
            onSaved();
        },
        bodyFormatter,
    });

    return (
        <Form validationBehavior={'native'} action={submitAction}>
            <Flex gap={'size-200'} direction={'column'}>
                <>{componentFields(state)}</>

                <Button type='submit' isDisabled={isPending} UNSAFE_style={{ maxWidth: 'fit-content' }}>
                    Add & Connect
                </Button>
            </Flex>
        </Form>
    );
};
