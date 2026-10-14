// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { v4 as uuid } from 'uuid';

import { getObjectFromFormData, SinkOutputFormats, WebhookHttpMethod, WebhookSinkConfig } from '../utils';

export type Pair = Record<Fields, string>;

export enum Fields {
    KEY = 'key',
    VALUE = 'value',
}

export const getPairsFromObject = (obj: Record<string, string>): Pair[] => {
    return Object.entries(obj).map(([key, value]) => ({ key, value }));
};

export const getWebhookInitialConfig = (project_id: string): WebhookSinkConfig => ({
    id: uuid(),
    name: 'Webhook sink',
    timeout: 10,
    project_id,
    sink_type: 'webhook',
    rate_limit: 1,
    webhook_url: '',
    http_method: WebhookHttpMethod.POST,
    output_formats: [],
    headers: {},
});

const parseRateLimit = (value: FormDataEntryValue | null): number | null => {
    if (!value || value === '') return null;
    return Number(value);
};

export const webhookBodyFormatter = (formData: FormData): WebhookSinkConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    headers: getObjectFromFormData(formData.getAll('headers-keys'), formData.getAll('headers-values')),
    timeout: Number(formData.get('timeout')),
    sink_type: 'webhook',
    rate_limit: parseRateLimit(formData.get('rate_limit')),
    webhook_url: String(formData.get('webhook_url')),
    http_method: formData.get('http_method') as WebhookHttpMethod,
    output_formats: formData.getAll('output_formats') as SinkOutputFormats,
    project_id: String(formData.get('project_id')),
});
