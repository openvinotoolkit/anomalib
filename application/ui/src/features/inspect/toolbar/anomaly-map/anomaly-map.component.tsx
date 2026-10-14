import { usePatchPipeline, usePipeline, useProjectIdentifier } from '@anomalib-studio/hooks';
import { Flex, Switch } from '@geti-ui/ui';

import classes from './anomaly-map.module.scss';

export const AnomalyMap = () => {
    const { projectId } = useProjectIdentifier();
    const patchPipeline = usePatchPipeline(projectId);
    const { data: pipeline } = usePipeline();

    const hasOverlay = pipeline?.overlay ?? false;
    const isPipelineStopped = pipeline?.status !== 'running';

    const handleChange = () => {
        patchPipeline.mutateAsync({ params: { path: { project_id: projectId } }, body: { overlay: !hasOverlay } });
    };

    return (
        <Flex>
            <Switch
                UNSAFE_className={classes.switch}
                onChange={handleChange}
                isSelected={hasOverlay}
                isDisabled={patchPipeline.isPending || isPipelineStopped}
            >
                Anomaly Map
            </Switch>
        </Flex>
    );
};
