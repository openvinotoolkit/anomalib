import { usePipeline, useProjectIdentifier } from '@anomalib-studio/hooks';
import { Flex } from '@geti-ui/ui';
import isEmpty from 'lodash-es/isEmpty';

import { StreamContainer } from '../stream-container/stream-container';
import { EnableProject } from './enable-project/enable-project.component';
import { useActivePipelineStatus } from './hooks/use-active-pipeline-status.hook';
import { SourceSinkMessage } from './source-sink-message/source-sink-message.component';

import classes from './main-content.module.scss';

export const MainContent = () => {
    const { data: pipeline } = usePipeline();
    const { projectId } = useProjectIdentifier();
    const { hasActiveProject, isCurrentProjectActive, activeProjectId } = useActivePipelineStatus(projectId);

    if (isEmpty(pipeline.source?.id)) {
        return <SourceSinkMessage />;
    }

    const showEnableProject = hasActiveProject && !isCurrentProjectActive;

    return (
        <>
            {showEnableProject && (
                <EnableProject currentProjectId={projectId} activeProjectId={String(activeProjectId)} />
            )}

            {!showEnableProject && (
                <Flex
                    width={'100%'}
                    height={'100%'}
                    gridArea={'canvas'}
                    maxHeight={'100%'}
                    alignItems={'center'}
                    justifyContent={'center'}
                    UNSAFE_className={classes.canvasContainer}
                >
                    <StreamContainer hasActiveProject={hasActiveProject} />
                </Flex>
            )}
        </>
    );
};
