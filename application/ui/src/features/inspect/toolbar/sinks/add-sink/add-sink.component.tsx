import { ReactNode } from 'react';

import { Button, Form } from '@geti-ui/ui';

import { useConnectSinkToPipeline } from '../../../../../hooks/use-pipeline.hook';
import { useSinkAction } from '../hooks/use-sink-action.hook';
import { SinkConfig } from '../utils';

interface AddSinkProps<T> {
    config: Awaited<T>;
    onSaved: () => void;
    componentFields: (state: Awaited<T>) => ReactNode;
    bodyFormatter: (formData: FormData) => T;
}

export const AddSink = <T extends SinkConfig>({ config, onSaved, bodyFormatter, componentFields }: AddSinkProps<T>) => {
    const connectToPipelineMutation = useConnectSinkToPipeline();

    const [state, submitAction, isPending] = useSinkAction({
        config,
        isNewSink: true,
        onSaved: async (sourceId) => {
            await connectToPipelineMutation(sourceId);
            onSaved();
        },
        bodyFormatter,
    });

    return (
        <Form validationBehavior={'native'} action={submitAction}>
            <>{componentFields(state)}</>

            <Button type='submit' isDisabled={isPending} UNSAFE_style={{ maxWidth: 'fit-content' }}>
                Add & Connect
            </Button>
        </Form>
    );
};
