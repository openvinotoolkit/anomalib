import { useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import { ActionButton, AlertDialog, DialogContainer, Item, Key, Menu, MenuTrigger } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

interface ProjectActionsProps {
    projectId: string;
    projectName: string;
    isLastProject: boolean;
    onRename: () => void;
}

const PROJECT_ACTIONS = {
    RENAME: 'Rename',
    DELETE: 'Delete',
} as const;

export const ProjectActions = ({ projectId, projectName, isLastProject, onRename }: ProjectActionsProps) => {
    const { projectId: currentProjectId } = useProjectIdentifier();
    const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

    const deleteProjectMutation = $api.useMutation('delete', '/api/projects/{project_id}', {
        meta: {
            invalidates: [['get', '/api/projects']],
        },
    });

    const isCurrentProject = projectId === currentProjectId;
    const canDeleteProject = !isLastProject && !isCurrentProject;

    const handleAction = (key: Key) => {
        if (key === PROJECT_ACTIONS.RENAME) {
            onRename();
        }
        if (key === PROJECT_ACTIONS.DELETE && canDeleteProject) {
            setIsDeleteDialogOpen(true);
        }
    };

    const handleDeleteProject = () => {
        deleteProjectMutation.mutate(
            { params: { path: { project_id: projectId } } },
            {
                onSuccess: () => {
                    toast({ type: 'success', message: `Project "${projectName}" has been deleted.` });
                    setIsDeleteDialogOpen(false);
                },
                onError: () => {
                    toast({ type: 'error', message: `Failed to delete project "${projectName}".` });
                    setIsDeleteDialogOpen(false);
                },
            }
        );
    };

    const getDisabledKeys = () => {
        const keys: Key[] = [];
        if (!canDeleteProject) {
            keys.push(PROJECT_ACTIONS.DELETE);
        }
        return keys;
    };

    return (
        <>
            <MenuTrigger>
                <ActionButton isQuiet aria-label='project actions' UNSAFE_className='actionMenu'>
                    <MoreMenu />
                </ActionButton>
                <Menu onAction={handleAction} disabledKeys={getDisabledKeys()}>
                    <Item key={PROJECT_ACTIONS.RENAME}>{PROJECT_ACTIONS.RENAME}</Item>
                    <Item key={PROJECT_ACTIONS.DELETE}>{PROJECT_ACTIONS.DELETE}</Item>
                </Menu>
            </MenuTrigger>

            <DialogContainer onDismiss={() => setIsDeleteDialogOpen(false)}>
                {isDeleteDialogOpen ? (
                    <AlertDialog
                        variant='destructive'
                        cancelLabel='Cancel'
                        title={`Delete project "${projectName}"?`}
                        primaryActionLabel={deleteProjectMutation.isPending ? 'Deleting...' : 'Delete'}
                        isPrimaryActionDisabled={deleteProjectMutation.isPending}
                        onPrimaryAction={handleDeleteProject}
                    >
                        Deleting a project will remove all associated data including images, models, and configurations.
                        This action cannot be undone.
                    </AlertDialog>
                ) : null}
            </DialogContainer>
        </>
    );
};
