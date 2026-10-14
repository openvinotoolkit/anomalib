// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    useActivateAndRunPipeline,
    useDisablePipeline,
    usePipeline,
    useProjectIdentifier,
} from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import { Switch } from '@geti-ui/ui';

export const TogglePipelineButton = () => {
    const pipelineQuery = usePipeline();

    const { projectId } = useProjectIdentifier();
    const disablePipelineMutation = useDisablePipeline(projectId);
    const activateAndRunPipeline = useActivateAndRunPipeline({
        onSuccess: () => toast({ type: 'success', message: `Pipeline enabled successfully` }),
    });

    const isPending = disablePipelineMutation.isPending || activateAndRunPipeline.isPending;
    const isPipelineEnabled = pipelineQuery.data?.status === 'running' || pipelineQuery.data?.status === 'active';

    const handleToggle = async () => {
        if (isPipelineEnabled) {
            disablePipelineMutation.mutate(
                { params: { path: { project_id: projectId } } },
                { onSuccess: () => toast({ type: 'success', message: `Pipeline disabled successfully` }) }
            );
        } else {
            await activateAndRunPipeline.mutateAsync();
        }
    };

    return (
        <Switch
            isEmphasized
            isSelected={isPipelineEnabled}
            isDisabled={isPending}
            onChange={handleToggle}
            UNSAFE_style={{ margin: '0px' }}
        >
            Pipeline {isPipelineEnabled ? 'enabled' : 'disabled'}
        </Switch>
    );
};
