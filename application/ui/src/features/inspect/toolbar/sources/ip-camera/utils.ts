import { v4 as uuid } from 'uuid';

import { IPCameraSourceConfig } from '../util';

export const getIpCameraInitialConfig = (projectId: string): IPCameraSourceConfig => ({
    id: uuid(),
    name: 'IP camera source',
    project_id: projectId,
    source_type: 'ip_camera',
    stream_url: '',
    auth_required: false,
});

export const ipCameraBodyFormatter = (formData: FormData): IPCameraSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'ip_camera',
    project_id: String(formData.get('project_id')),
    stream_url: String(formData.get('stream_url')),
    auth_required: String(formData.get('auth_required')) === 'on',
});
