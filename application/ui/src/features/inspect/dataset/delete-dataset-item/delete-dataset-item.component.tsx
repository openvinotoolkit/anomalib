// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import { ActionButton, DialogContainer } from '@geti-ui/ui';
import { Delete } from '@geti-ui/ui/icons';
import { isFunction } from 'lodash-es';
import { useOverlayTriggerState } from 'react-stately';

import { $api } from '../../../../api/client';
import { AlertDialogContent } from './alert-dialog-content.component';

import classes from './delete-dataset-item.module.scss';

export interface DeleteMediaItemProps {
    itemsIds: string[];
    onDeleted?: (deletedIds: string[]) => void;
}

const isFulfilled = (response: PromiseSettledResult<{ itemId: string }>) => response.status === 'fulfilled';

export const DeleteMediaItem = ({ itemsIds = [], onDeleted }: DeleteMediaItemProps) => {
    const alertDialogState = useOverlayTriggerState({});
    const { projectId: project_id } = useProjectIdentifier();

    const removeMutation = $api.useMutation('delete', `/api/projects/{project_id}/images/{media_id}`, {
        meta: { invalidates: [['get', '/api/projects/{project_id}/images', { params: { path: { project_id } } }]] },
        onError: (error, { params: { path } }) => {
            const { media_id: itemId } = path;

            toast({
                id: itemId,
                type: 'error',
                message: `Failed to delete, ${error?.detail}`,
            });
        },
    });

    const handleRemoveItems = async () => {
        alertDialogState.close();

        toast({ id: 'deleting-notification', type: 'info', message: `Deleting items...` });

        const deleteItemPromises = itemsIds.map(async (media_id) => {
            await removeMutation.mutateAsync({ params: { path: { project_id, media_id } } });

            return { itemId: media_id };
        });

        const responses = await Promise.allSettled(deleteItemPromises);
        const deletedIds = responses.filter(isFulfilled).map(({ value }) => value.itemId);

        isFunction(onDeleted) && onDeleted(deletedIds);

        toast({
            id: 'deleting-notification',
            type: 'success',
            message: `${deletedIds.length} item(s) deleted successfully`,
            duration: 3000,
        });
    };

    return (
        <>
            <ActionButton
                isQuiet
                aria-label='delete media item'
                onPress={alertDialogState.open}
                isDisabled={removeMutation.isPending}
                UNSAFE_className={classes.deleteButton}
            >
                <Delete />
            </ActionButton>

            <DialogContainer onDismiss={alertDialogState.close}>
                {alertDialogState.isOpen && (
                    <AlertDialogContent itemsIds={itemsIds} onPrimaryAction={handleRemoveItems} />
                )}
            </DialogContainer>
        </>
    );
};
