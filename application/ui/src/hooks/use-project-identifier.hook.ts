/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useParams } from 'react-router';

export const useProjectIdentifier = () => {
    const { projectId } = useParams();

    if (projectId === undefined) {
        throw new Error('Project ID is not defined');
    }

    return { projectId };
};
