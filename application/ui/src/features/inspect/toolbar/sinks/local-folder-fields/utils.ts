// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { v4 as uuid } from 'uuid';

import { LocalFolderSinkConfig, SinkOutputFormats } from '../utils';

export const getLocalFolderInitialConfig = (project_id: string): LocalFolderSinkConfig => ({
    id: uuid(),
    name: 'Local folder sink',
    project_id,
    sink_type: 'folder',
    rate_limit: 1,
    folder_path: '', // Only the suffix after {datapath}/sinks/
    output_formats: [],
});

const parseRateLimit = (value: FormDataEntryValue | null): number | null => {
    if (!value || value === '') return null;
    return Number(value);
};

export const localFolderBodyFormatter = (formData: FormData): LocalFolderSinkConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    sink_type: 'folder',
    rate_limit: parseRateLimit(formData.get('rate_limit')),
    folder_path: String(formData.get('folder_path')),
    project_id: String(formData.get('project_id')),
    output_formats: formData.getAll('output_formats') as SinkOutputFormats,
});
