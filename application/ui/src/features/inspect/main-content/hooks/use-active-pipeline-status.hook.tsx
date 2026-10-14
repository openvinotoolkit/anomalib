// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useActivePipeline } from '@anomalib-studio/hooks';
import isEmpty from 'lodash-es/isEmpty';

export const useActivePipelineStatus = (projectId: string) => {
    const { data: activeProjectPipeline } = useActivePipeline();

    const hasActiveProject = !isEmpty(activeProjectPipeline?.project_id);
    const isCurrentProjectActive = activeProjectPipeline?.project_id === projectId;

    return { hasActiveProject, isCurrentProjectActive, activeProjectId: activeProjectPipeline?.project_id };
};
