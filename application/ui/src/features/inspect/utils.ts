import { getApiUrl } from '@anomalib-studio/api';
import { isTauri } from '@tauri-apps/api/core';
import { isString } from 'lodash-es';

import { MediaItem } from './dataset/types';

const pluralRules = new Intl.PluralRules('en');

export const removeUnderscore = (text: string) => {
    return text.replaceAll('_', ' ');
};

export const isStatusActive = (status: string) => {
    return ['running', 'active'].includes(status);
};

/**
 * Tauri save dialog extensions for downloads.
 * Covers: compressed files, images, OpenVINO (xml/bin), ONNX, Torch (pt/pth).
 * Using concrete extensions instead of '*' avoids platform/runtime issues.
 */
const TauriSaveDialogExtensions = [
    // Compressed (model exports, logs, etc.)
    'zip',
    'tar',
    'gz',
    'tgz',
    '7z',
    // Images (dataset downloads)
    'jpg',
    'jpeg',
    'png',
    'gif',
    'bmp',
    'webp',
    'tiff',
    'tif',
    'jfif',
    // OpenVINO model
    'xml',
    'bin',
    // ONNX model
    'onnx',
    // Torch model
    'pt',
    'pth',
] as const;

/**
 * Downloads a blob as a file. In Tauri, shows a save dialog to let the user choose the location.
 * In browser, uses the traditional download approach.
 */
export const downloadBlob = async (blob: Blob, filename: string): Promise<void> => {
    if (isTauri()) {
        try {
            const { save } = await import('@tauri-apps/plugin-dialog');
            const { writeFile } = await import('@tauri-apps/plugin-fs');

            const filePath = await save({
                defaultPath: filename,
                filters: [
                    {
                        name: 'Supported types',
                        extensions: [...TauriSaveDialogExtensions],
                    },
                ],
            });

            if (filePath) {
                // Convert blob to Uint8Array and write to file
                const arrayBuffer = await blob.arrayBuffer();
                const contents = new Uint8Array(arrayBuffer);
                await writeFile(filePath, contents);
            }
        } catch {
            // Fallback to browser download if Tauri save fails
            downloadBlobBrowser(blob, filename);
        }
    } else {
        downloadBlobBrowser(blob, filename);
    }
};

/**
 * Browser-based blob download using anchor element
 */
const downloadBlobBrowser = (blob: Blob, filename: string): void => {
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
};

export const sanitizeFilename = (name: string): string => {
    return name
        .replace(/\s+/g, '_')
        .replace(/[^a-zA-Z0-9_\-\.]/g, '')
        .toLowerCase();
};

export const formatSize = (bytes: number | null | undefined) => {
    if (bytes === null || bytes === undefined) {
        return '';
    }

    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex += 1;
    }

    const maximumFractionDigits = size >= 10 ? 0 : 1;
    const formatter = new Intl.NumberFormat(undefined, { maximumFractionDigits });

    return `${formatter.format(size)} ${units[unitIndex]}`;
};

export const isNonEmptyString = (value: unknown): value is string => isString(value) && value !== '';

export const getThumbnailUrl = (mediaItem: MediaItem) =>
    getApiUrl(`/api/projects/${mediaItem.project_id}/images/${mediaItem.id}/thumbnail`);

export const formatDuration = (seconds: number | null): string | null => {
    if (seconds === null) return null;

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;

    if (hours > 0) {
        return `${hours}h ${minutes}m`;
    }
    if (minutes > 0) {
        return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
};

export const pluralize = (count: number, singular: string, plural: string): string =>
    pluralRules.select(count) === 'one' ? singular : plural;
