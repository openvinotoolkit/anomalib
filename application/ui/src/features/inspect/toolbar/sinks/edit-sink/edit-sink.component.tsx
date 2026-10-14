import { ReactNode, useRef } from 'react';

import { ActionButton, Button, ButtonGroup, Divider, Flex, Form, Text, View } from '@geti-ui/ui';
import { Back } from '@geti-ui/ui/icons';

import { useConnectSinkToPipeline } from '../../../../../hooks/use-pipeline.hook';
import { useSinkAction } from '../hooks/use-sink-action.hook';
import { SinkConfig } from '../utils';

import classes from './edit-sink.module.scss';

interface EditSinkProps<T> {
    config: Awaited<T>;
    onSaved: () => void;
    onBackToList: () => void;
    componentFields: (state: Awaited<T>) => ReactNode;
    bodyFormatter: (formData: FormData) => T;
}

export const EditSink = <T extends SinkConfig>({
    config,
    onSaved,
    onBackToList,
    bodyFormatter,
    componentFields,
}: EditSinkProps<T>) => {
    const connectToPipeline = useRef(false);
    const connectToPipelineMutation = useConnectSinkToPipeline();

    const [state, submitAction, isPending] = useSinkAction({
        config,
        isNewSink: false,
        onSaved: async (sinkId) => {
            connectToPipeline.current && (await connectToPipelineMutation(sinkId));
            connectToPipeline.current = false;
            onSaved();
        },
        bodyFormatter,
    });

    return (
        <Form validationBehavior={'native'} action={submitAction}>
            <Flex gap={'size-100'} alignItems={'center'} marginTop={'0px'} justifyContent={'space-between'}>
                <ActionButton isQuiet onPress={onBackToList}>
                    <Back />
                </ActionButton>

                <Text>Edit sink</Text>
            </Flex>

            <View UNSAFE_className={classes.container}>
                <>{componentFields(state)}</>
            </View>
            <Divider size='S' marginY={'size-200'} />

            <ButtonGroup marginTop={'0px'}>
                <Button
                    type='submit'
                    isDisabled={isPending}
                    UNSAFE_style={{ maxWidth: 'fit-content' }}
                    onPress={() => (connectToPipeline.current = false)}
                >
                    Save
                </Button>

                <Button
                    type='submit'
                    isDisabled={isPending}
                    UNSAFE_style={{ maxWidth: 'fit-content' }}
                    onPress={() => (connectToPipeline.current = true)}
                >
                    Save & Connect
                </Button>
            </ButtonGroup>
        </Form>
    );
};
