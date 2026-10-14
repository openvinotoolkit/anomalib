import { waitFor } from '@testing-library/react';
import { HttpResponse } from 'msw';
import { SchemaPredictionResponse } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';
import { queryClient } from 'src/query-client/query-client';

import { getMockedMediaItem } from '../../../../../../mocks/mock-media-item';
import { getMockedPipeline } from '../../../../../../mocks/mock-pipeline';
import { renderHook } from '../../../../../../tests/utils';
import { useMediaItemInference } from './use-media-item-inference.hook';

vi.mock('./util', () => ({
    downloadImageAsFile: () => new Blob(['fake-image-data'], { type: 'image/jpeg' }),
}));

const mockedModel = {
    id: 'model-123',
    name: 'Test Model (model-123)',
    architecture: 'padim',
    format: 'onnx' as const,
    project_id: 'project-123',
    threshold: 0.5,
    is_ready: true,
    train_job_id: 'train-job-1',
    dataset_snapshot_id: '',
};

describe('useMediaItemInference', () => {
    const mockMediaItem = getMockedMediaItem({ id: 'media-123', project_id: 'project-123', filename: 'test.jpg' });
    const mockPrediction: SchemaPredictionResponse = {
        score: 0.95,
        label: 'Anomalous',
        anomaly_map: 'base64-encoded-image',
    };

    const renderMediaItemInference = ({
        mediaItem = mockMediaItem,
        pipelineConfig = {},
    }: {
        mediaItem?: typeof mockMediaItem;
        pipelineConfig?: Partial<ReturnType<typeof getMockedPipeline>> | null;
    } = {}) => {
        server.use(
            http.post('/api/projects/{project_id}/models/{model_id}:predict', () => HttpResponse.json(mockPrediction)),
            http.get('/api/projects/{project_id}/pipeline', () => {
                if (pipelineConfig === null) {
                    return HttpResponse.json(null);
                }
                return HttpResponse.json(getMockedPipeline(pipelineConfig));
            })
        );

        return renderHook(() => useMediaItemInference(mediaItem));
    };

    beforeEach(() => {
        vi.clearAllMocks();
        queryClient.clear();
    });

    it('fetches inference result when model is available', async () => {
        const { result } = renderMediaItemInference({ pipelineConfig: { model: mockedModel } });

        await waitFor(() => {
            expect(result.current.isSuccess).toBe(true);
        });

        expect(result.current.data).toEqual(mockPrediction);
    });

    it('does not fetch inference when no model is configured', async () => {
        const { result } = renderMediaItemInference({ pipelineConfig: { model: undefined } });

        await waitFor(() => {
            expect(result.current.isPending).toBe(true);
        });

        expect(result.current.data).toBeUndefined();
        expect(result.current.fetchStatus).toBe('idle');
    });

    it('does not fetch inference when model id is empty', async () => {
        const { result } = renderMediaItemInference({ pipelineConfig: { model: { ...mockedModel, id: '' } } });

        await waitFor(() => {
            expect(result.current.isPending).toBe(true);
        });

        expect(result.current.data).toBeUndefined();
        expect(result.current.fetchStatus).toBe('idle');
    });

    it('does not fetch inference when pipeline is not loaded', async () => {
        const { result } = renderMediaItemInference({ pipelineConfig: null });

        await waitFor(() => {
            expect(result.current.isPending).toBe(true);
        });

        expect(result.current.data).toBeUndefined();
        expect(result.current.fetchStatus).toBe('idle');
    });
});
