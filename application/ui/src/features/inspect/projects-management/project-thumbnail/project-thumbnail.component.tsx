// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { $api, getApiUrl } from '@anomalib-studio/api';
import { Image, PhotoPlaceholder } from '@geti-ui/ui';

import { isNonEmptyString } from '../../utils';

interface ProjectThumbnailProps {
    projectId?: string;
    projectName: string;
    size?: string;
}

const ProjectImage = ({ projectId, projectName, size }: ProjectThumbnailProps & { projectId: string }) => {
    const { data: mediaList } = $api.useQuery('get', '/api/projects/{project_id}/images', {
        params: { path: { project_id: projectId }, query: { limit: 1, offset: 0 } },
    });

    const firstMedia = mediaList?.media?.[0];

    if (isNonEmptyString(firstMedia?.id)) {
        return (
            <Image
                src={getApiUrl(`/api/projects/${projectId}/images/${firstMedia.id}/thumbnail`)}
                alt={`${projectName} thumbnail`}
                height={size}
                width={size}
                objectFit='cover'
                UNSAFE_style={{ borderRadius: '50%' }}
            />
        );
    }
    return <PhotoPlaceholder name={projectName} indicator={projectId} height={size} width={size} />;
};

export const ProjectThumbnail = ({ projectId, projectName, size = 'size-300' }: ProjectThumbnailProps) => {
    if (projectId) {
        return <ProjectImage projectId={projectId} projectName={projectName} size={size} />;
    }
    return <PhotoPlaceholder name={projectName} indicator={projectId ?? projectName} height={size} width={size} />;
};
