import { useActionState } from 'react';

import { toast } from '@anomalib-studio/toast';
import { isFunction } from 'lodash-es';

import { SinkConfig } from '../utils';
import { useSinkMutation } from './use-sink-mutation.hook';

interface useSinkActionProps<T> {
    config: Awaited<T>;
    isNewSink: boolean;
    onSaved?: (sink_id: string) => void;
    bodyFormatter: (formData: FormData) => T;
}

export const useSinkAction = <T extends SinkConfig>({
    config,
    isNewSink,
    onSaved,
    bodyFormatter,
}: useSinkActionProps<T>) => {
    const addOrUpdateSink = useSinkMutation(isNewSink);

    return useActionState<T, FormData>(async (_prevState: T, formData: FormData) => {
        const body = bodyFormatter(formData);

        try {
            const sink_id = await addOrUpdateSink(body);

            toast({
                type: 'success',
                message: `Sink configuration ${isNewSink ? 'created' : 'updated'} successfully.`,
            });

            isFunction(onSaved) && onSaved(sink_id);

            return { ...body, id: sink_id };
        } catch (error: unknown) {
            const details = (error as { detail?: string })?.detail;

            toast({
                type: 'error',
                message: `Failed to save sink configuration, ${details ?? 'please try again'}`,
            });
        }

        return body;
    }, config);
};
