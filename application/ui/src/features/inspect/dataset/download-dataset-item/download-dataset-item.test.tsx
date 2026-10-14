// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ThemeProvider } from '@geti-ui/ui';
import { fireEvent, render, screen } from '@testing-library/react';
import { getMockedMediaItem } from 'mocks/mock-media-item';

import { MediaItem } from '../types';
import { DownloadDatasetItem } from './download-dataset-item.component';
import { downloadFile } from './utils';

vi.mock('./utils', async (importActual) => {
    const actual = await importActual<typeof import('./utils')>();
    return {
        ...actual,
        downloadFile: vi.fn(),
    };
});

describe('DownloadDatasetItem', () => {
    const renderComponent = (mediaItem: MediaItem) => {
        return render(
            <ThemeProvider>
                <DownloadDatasetItem mediaItem={mediaItem} />
            </ThemeProvider>
        );
    };

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('calls downloadFile with correct URL and filename when button is clicked', () => {
        const mockMediaItem = getMockedMediaItem({});

        renderComponent(mockMediaItem);

        const button = screen.getByLabelText('download media item');
        fireEvent.click(button);

        expect(downloadFile).toHaveBeenCalledTimes(1);
        expect(downloadFile).toHaveBeenCalledWith(
            `/api/projects/${mockMediaItem.project_id}/images/${mockMediaItem.id}/full`,
            'test-image.jpg'
        );
    });
});
