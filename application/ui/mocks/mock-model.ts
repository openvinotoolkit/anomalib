import { SchemaModel } from 'src/api/openapi-spec';
import { ModelData } from 'src/hooks/utils';

export const getMockedModel = (partial: Partial<SchemaModel>): SchemaModel => ({
    id: 'model-1',
    name: 'Model 1',
    format: 'openvino' as const,
    project_id: '123',
    threshold: 0.5,
    is_ready: true,
    dataset_snapshot_id: 'dataset-1',
    train_job_id: 'job-1',
    architecture: 'PatchCore',
    ...partial,
});

export const getMockedModelData = (overrides: Partial<ModelData> = {}): ModelData => ({
    id: 'model-1',
    name: 'PatchCore',
    status: 'Completed',
    architecture: 'PatchCore',
    timestamp: 'Dec 18, 2025, 10:30 AM',
    backbone: 'resnet50',
    startTime: Date.now() - 3600000,
    progress: 100,
    durationInSeconds: 120,
    sizeBytes: 52428800,
    job: undefined,
    ...overrides,
});
