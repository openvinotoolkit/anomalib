import { useActivePipeline } from '@anomalib-studio/hooks';
import isEmpty from 'lodash-es/isEmpty';

export const useEnsureActivePipeline = (projectId: string) => {
    const { data: activeProjectPipeline } = useActivePipeline();

    const hasActiveProject = !isEmpty(activeProjectPipeline?.project_id);
    const isCurrentProjectActive = activeProjectPipeline?.project_id === projectId;

    return { hasActiveProject, isCurrentProjectActive, activeProjectId: activeProjectPipeline?.project_id };
};
