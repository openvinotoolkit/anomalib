import { SchemaPipeline } from './../src/api/openapi-spec.d';

export const getMockedPipeline = (customPipeline?: Partial<SchemaPipeline>): SchemaPipeline => {
    return {
        project_id: '123',
        status: 'running' as const,
        source: {
            id: 'source-id',
            name: 'source',
            project_id: '123',
            source_type: 'video_file' as const,
            video_path: 'video.mp4',
        },
        model: {
            id: '1',
            name: 'Object_Detection_TestModel (1)',
            format: 'onnx' as const,
            project_id: '123',
            threshold: 0.5,
            is_ready: true,
            train_job_id: 'train-job-1',
            dataset_snapshot_id: '',
            architecture: 'PatchCore',
        },
        sink: {
            id: 'sink-id',
            name: 'sink',
            project_id: '123',
            folder_path: 'data/sink',
            output_formats: ['image_original', 'image_with_predictions', 'predictions'] as Array<
                'image_original' | 'image_with_predictions' | 'predictions'
            >,
            rate_limit: 0.2,
            sink_type: 'folder' as const,
        },
        inference_device: 'CPU',
        ...customPipeline,
    };
};
