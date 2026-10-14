import { $api } from '@anomalib-studio/api';
import {
    useActivateAndRunPipeline,
    useActivatePipeline,
    useDisablePipeline,
    usePipeline,
} from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import { AlertDialog } from '@geti-ui/ui';
import { useIsPipelineConfigured } from 'src/hooks/use-is-pipeline-configured.hook';

interface ConfirmationDialogProps {
    activeProjectId: string;
    currentProjectId: string;
}

export const ConfirmationDialog = ({ activeProjectId, currentProjectId }: ConfirmationDialogProps) => {
    const { data: pipeline } = usePipeline();
    const activatePipeline = useActivatePipeline({});
    const disablePipeline = useDisablePipeline(activeProjectId);
    const canEnablePipeline = useIsPipelineConfigured(pipeline);

    const activateAndRunPipeline = useActivateAndRunPipeline({
        onSuccess: () => toast({ type: 'success', message: `Pipeline enabled successfully` }),
    });

    const activeProject = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: activeProjectId } },
    });

    const currentProject = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: currentProjectId } },
    });

    const isUpdating =
        activeProject.isLoading ||
        currentProject.isLoading ||
        disablePipeline.isPending ||
        activatePipeline.isPending ||
        activateAndRunPipeline.isPending;

    const handleEnableProject = async () => {
        await disablePipeline.mutateAsync({
            params: { path: { project_id: activeProjectId } },
        });

        if (canEnablePipeline) {
            await activateAndRunPipeline.mutateAsync();
        } else {
            await activatePipeline.mutateAsync({
                params: { path: { project_id: currentProjectId } },
            });
        }
    };

    return (
        <AlertDialog
            variant={'confirmation'}
            cancelLabel={'Cancel'}
            onPrimaryAction={handleEnableProject}
            primaryActionLabel={isUpdating ? 'Activating...' : 'Activate project'}
            isPrimaryActionDisabled={isUpdating}
            title={`Activate project "${currentProject.data.name}"`}
        >
            By activating this project, the current active project &quot;{activeProject.data.name}&quot; and its
            respective inference workflow will be disabled. Changing your active project can influence any downstream
            processes that might be connected to the output generated from the current active project.
        </AlertDialog>
    );
};
