// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { SchemaPipeline } from 'src/api/openapi-spec';
import { isNonEmptyString } from 'src/features/inspect/utils';

export const useIsPipelineConfigured = (pipeline?: SchemaPipeline) => {
    if (!pipeline) return false;

    const { model, source } = pipeline;

    return isNonEmptyString(model?.id) && isNonEmptyString(source?.id);
};
