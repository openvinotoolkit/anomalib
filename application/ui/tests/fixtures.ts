import { createNetworkFixture, NetworkFixture } from '@msw/playwright';
import { expect, test as testBase } from '@playwright/test';

import { getOpenApiHttp, handlers } from '../src/api/utils';

// Playwright component tests serve the app at localhost:3000 with relative API URLs,
// so MSW handlers need an empty baseUrl to match requests like /api/projects.
const http = getOpenApiHttp('');

interface Fixtures {
    network: NetworkFixture;
}

const test = testBase.extend<Fixtures>({
    network: createNetworkFixture({
        initialHandlers: [
            http.get('/api/projects', ({ response }) => {
                return response(200).json({
                    projects: [
                        {
                            id: '12',
                            name: 'Project #12',
                        },
                    ],
                    pagination: { offset: 0, limit: 20, count: 1, total: 1 },
                });
            }),
            http.get('/api/projects/{project_id}', ({ response }) => {
                return response(200).json({
                    id: '1',
                    name: 'Project #1',
                });
            }),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) => {
                return response(200).json({
                    status: 'idle',
                    project_id: '12',
                });
            }),
            http.get('/api/projects/{project_id}/images', ({ response }) => {
                return response(200).json({ media: [] });
            }),
            http.get('/api/system/devices/inference', ({ response }) => {
                return response(200).json([
                    { type: 'cpu', name: 'CPU', memory: null, index: null, openvino_name: 'CPU' },
                ]);
            }),
            http.get('/api/projects/{project_id}/models', ({ response }) => {
                return response(200).json({ models: [] });
            }),
            http.get('/api/active-pipeline', ({ response }) => {
                return response(200).json({ project_id: '12', status: 'idle' });
            }),
            http.get('/api/jobs', ({ response }) => {
                return response(200).json({ jobs: [], pagination: { total: 0, skip: 0, limit: 20 } });
            }),
            // Auto-generated handlers from the OpenAPI spec as fallback
            ...handlers,
        ],
    }),
});

export { expect, http, test };
