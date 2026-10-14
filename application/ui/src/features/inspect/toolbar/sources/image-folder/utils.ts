import { v4 as uuid } from 'uuid';

import { ImagesFolderSourceConfig } from '../util';

export const getImageFolderInitialConfig = (projectId: string): ImagesFolderSourceConfig => ({
    id: uuid(),
    name: 'Images folder source',
    project_id: projectId,
    source_type: 'images_folder',
    images_folder_path: '',
    ignore_existing_images: false,
});

export const imageFolderBodyFormatter = (formData: FormData): ImagesFolderSourceConfig => ({
    id: String(formData.get('id')),
    name: String(formData.get('name')),
    source_type: 'images_folder',
    project_id: String(formData.get('project_id')),
    images_folder_path: String(formData.get('images_folder_path')),
    ignore_existing_images: formData.get('ignore_existing_images') === 'on',
});
