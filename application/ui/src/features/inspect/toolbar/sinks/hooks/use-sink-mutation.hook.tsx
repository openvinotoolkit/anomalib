import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { omit } from 'lodash-es';

import { SinkConfig } from '../utils';

export const useSinkMutation = (isNewSink: boolean) => {
    const { projectId } = useProjectIdentifier();
    const addSink = $api.useMutation('post', '/api/projects/{project_id}/sinks', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/sinks', { params: { path: { project_id: projectId } } }]],
        },
    });
    const updateSink = $api.useMutation('patch', '/api/projects/{project_id}/sinks/{sink_id}', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/sinks', { params: { path: { project_id: projectId } } }]],
        },
    });

    return async (body: SinkConfig) => {
        if (isNewSink) {
            // Omit id and project_id when creating - they're auto-generated/injected from URL
            const sinkPayload = omit(body, ['id', 'project_id']) as Parameters<typeof addSink.mutateAsync>[0]['body'];

            const response = await addSink.mutateAsync({
                body: sinkPayload,
                params: { path: { project_id: projectId } },
            });

            return String(response.id);
        }

        const response = await updateSink.mutateAsync({
            params: { path: { project_id: projectId, sink_id: String(body.id) } },
            body: omit(body, 'sink_type'),
        });

        return String(response.id);
    };
};
