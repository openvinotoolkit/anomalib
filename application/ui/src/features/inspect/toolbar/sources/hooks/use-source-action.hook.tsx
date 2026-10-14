// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActionState } from 'react';

import { toast } from '@anomalib-studio/toast';
import { isFunction } from 'lodash-es';

import { SourceConfig } from '../util';
import { useSourceMutation } from './use-source-mutation.hook';

interface useSourceActionProps<T> {
    config: Awaited<T>;
    isNewSource: boolean;
    onSaved?: (source_id: string) => void;
    bodyFormatter: (formData: FormData) => T;
}

export const useSourceAction = <T extends SourceConfig>({
    config,
    isNewSource,
    onSaved,
    bodyFormatter,
}: useSourceActionProps<T>) => {
    const addOrUpdateSource = useSourceMutation(isNewSource);

    return useActionState<T, FormData>(async (_prevState: T, formData: FormData) => {
        const body = bodyFormatter(formData);

        try {
            const source_id = await addOrUpdateSource(body);

            toast({
                type: 'success',
                message: `Source configuration ${isNewSource ? 'created' : 'updated'} successfully.`,
            });

            isFunction(onSaved) && onSaved(source_id);
            return { ...body, id: source_id };
        } catch (error: unknown) {
            const details = (error as { detail?: string })?.detail;

            toast({
                type: 'error',
                message: `Failed to save source configuration. ${details ?? 'Please try again'}`,
            });
        }

        return body;
    }, config);
};
