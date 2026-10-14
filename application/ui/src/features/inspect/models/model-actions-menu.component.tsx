import { useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { usePatchPipeline, useProjectIdentifier } from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import { ActionButton, AlertDialog, DialogContainer, Item, Menu, MenuTrigger, type Key } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

import type { ModelData } from '../../../hooks/utils';
import { JobLogsDialog } from '../jobs/show-job-logs.component';

interface ModelActionsMenuProps {
    model: ModelData;
    selectedModelId: string | undefined;
}

type DialogType = 'logs' | 'delete' | 'activate' | null;

export const ModelActionsMenu = ({ model, selectedModelId }: ModelActionsMenuProps) => {
    const { projectId } = useProjectIdentifier();
    const patchPipeline = usePatchPipeline(projectId);
    const [openDialog, setOpenDialog] = useState<DialogType>(null);

    const cancelJobMutation = $api.useMutation('post', '/api/jobs/{job_id}:cancel');
    const deleteModelMutation = $api.useMutation('delete', '/api/projects/{project_id}/models/{model_id}', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/models', { params: { path: { project_id: projectId } } }],
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
                ['get', '/api/jobs'],
            ],
        },
    });
    const deleteJobMutation = $api.useMutation('delete', '/api/jobs/{job_id}', {
        meta: {
            invalidates: [['get', '/api/jobs']],
        },
    });

    const hasJobActions = Boolean(model.job?.id);
    const hasCompletedStatus = model.status === 'Completed';
    const canDeleteModel = hasCompletedStatus && model.id !== selectedModelId;
    const canDeleteJob =
        model.status !== 'Completed' && (model.job?.status === 'failed' || model.job?.status === 'canceled');
    const shouldShowMenu = hasJobActions || canDeleteModel || canDeleteJob;

    if (!shouldShowMenu) {
        return null;
    }

    const disabledMenuKeys: Key[] = [];
    if (cancelJobMutation.isPending) {
        disabledMenuKeys.push('cancel');
    }
    if (deleteModelMutation.isPending) {
        disabledMenuKeys.push('delete');
    }
    if (deleteJobMutation.isPending) {
        disabledMenuKeys.push('delete-job');
    }
    if (model.id === selectedModelId || patchPipeline.isPending) {
        disabledMenuKeys.push('activate');
    }

    const handleCancelJob = () => {
        if (!model.job?.id) {
            return;
        }

        void cancelJobMutation.mutateAsync(
            { params: { path: { job_id: model.job.id } } },
            {
                onError: () => {
                    toast({ type: 'error', message: 'Failed to cancel training job.' });
                },
            }
        );
    };

    const handleDeleteModel = () => {
        void deleteModelMutation.mutateAsync(
            { params: { path: { project_id: projectId, model_id: model.id } } },
            {
                onSuccess: () => {
                    if (selectedModelId === model.id) {
                        patchPipeline.mutateAsync({
                            params: { path: { project_id: projectId } },
                            body: { model_id: undefined },
                        });
                    }

                    toast({ type: 'success', message: `Model "${model.name}" has been deleted.` });
                },
                onError: () => {
                    toast({ type: 'error', message: `Failed to delete "${model.name}".` });
                },
                onSettled: () => {
                    setOpenDialog(null);
                },
            }
        );
    };

    const handleDeleteJob = () => {
        if (!model.job?.id) {
            return;
        }

        void deleteJobMutation.mutateAsync(
            { params: { path: { job_id: model.job.id } } },
            {
                onSuccess: () => {
                    toast({ type: 'success', message: 'Job record has been deleted.' });
                },
                onError: () => {
                    toast({ type: 'error', message: 'Failed to delete job record.' });
                },
            }
        );
    };

    const handleActivateModel = () => {
        void patchPipeline.mutateAsync(
            { params: { path: { project_id: projectId } }, body: { model_id: model.id } },
            {
                onSuccess: () => {
                    toast({ type: 'success', message: `Model "${model.name}" is now active.` });
                },
                onError: () => {
                    toast({ type: 'error', message: `Failed to activate "${model.name}".` });
                },
                onSettled: () => {
                    setOpenDialog(null);
                },
            }
        );
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label='model actions'>
                    <MoreMenu />
                </ActionButton>
                <Menu
                    disabledKeys={disabledMenuKeys}
                    onAction={(actionKey) => {
                        if (actionKey === 'logs' && model.job?.id) {
                            setOpenDialog('logs');
                        }
                        if (actionKey === 'cancel' && model.job?.id) {
                            void handleCancelJob();
                        }
                        if (actionKey === 'delete' && canDeleteModel) {
                            setOpenDialog('delete');
                        }
                        if (actionKey === 'delete-job' && canDeleteJob) {
                            void handleDeleteJob();
                        }
                        if (actionKey === 'activate' && hasCompletedStatus && model.id !== selectedModelId) {
                            setOpenDialog('activate');
                        }
                    }}
                >
                    {hasCompletedStatus ? <Item key='activate'>Activate</Item> : null}
                    {hasJobActions ? <Item key='logs'>View logs</Item> : null}
                    {model.job?.status === 'pending' || model.job?.status === 'running' ? (
                        <Item key='cancel'>Cancel training</Item>
                    ) : null}
                    {canDeleteModel ? <Item key='delete'>Delete model</Item> : null}
                    {canDeleteJob ? <Item key='delete-job'>Delete</Item> : null}
                </Menu>
            </MenuTrigger>

            <DialogContainer type='fullscreen' onDismiss={() => setOpenDialog(null)}>
                {openDialog === 'logs' && model.job?.id ? (
                    <JobLogsDialog close={() => setOpenDialog(null)} jobId={model.job.id} />
                ) : null}
            </DialogContainer>

            <DialogContainer onDismiss={() => setOpenDialog(null)}>
                {openDialog === 'delete' && canDeleteModel ? (
                    <AlertDialog
                        variant='destructive'
                        cancelLabel='Cancel'
                        title={`Delete model "${model.name}"?`}
                        primaryActionLabel={deleteModelMutation.isPending ? 'Deleting...' : 'Delete model'}
                        isPrimaryActionDisabled={deleteModelMutation.isPending}
                        onPrimaryAction={() => {
                            handleDeleteModel();
                        }}
                    >
                        Deleting a model removes any exported artifacts and cannot be undone.
                    </AlertDialog>
                ) : null}
            </DialogContainer>

            <DialogContainer onDismiss={() => setOpenDialog(null)}>
                {openDialog === 'activate' && hasCompletedStatus ? (
                    <AlertDialog
                        variant='confirmation'
                        cancelLabel='Cancel'
                        title={`Activate model "${model.name}"?`}
                        primaryActionLabel={patchPipeline.isPending ? 'Activating...' : 'Activate'}
                        isPrimaryActionDisabled={patchPipeline.isPending}
                        onPrimaryAction={() => {
                            handleActivateModel();
                        }}
                    >
                        This model will be used for inference in the pipeline. The current active model will be
                        replaced.
                    </AlertDialog>
                ) : null}
            </DialogContainer>
        </>
    );
};
