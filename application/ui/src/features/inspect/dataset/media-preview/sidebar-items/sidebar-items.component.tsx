// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Selection, View } from '@geti-ui/ui';
import { GridLayoutOptions } from 'react-aria-components';
import { getThumbnailUrl, isNonEmptyString } from 'src/features/inspect/utils';

import { GridMediaItem } from '../../../../..//components/virtualizer-grid-layout/grid-media-item/grid-media-item.component';
import { MediaThumbnail } from '../../../../../components/media-thumbnail/media-thumbnail.component';
import { VirtualizerGridLayout } from '../../../../../components/virtualizer-grid-layout/virtualizer-grid-layout.component';
import { DeleteMediaItem } from '../../delete-dataset-item/delete-dataset-item.component';
import { DownloadDatasetItem } from '../../download-dataset-item/download-dataset-item.component';
import { MediaItem } from '../../types';

interface SidebarItemsProps {
    mediaItems: MediaItem[];
    hasNextPage: boolean;
    isLoadingMore: boolean;
    selectedMediaItem: MediaItem;
    layoutOptions: GridLayoutOptions;
    loadMore: () => void;
    onSelectedMediaItem: (mediaItem: string | null) => Promise<URLSearchParams>;
}

export const SidebarItems = ({
    mediaItems,
    hasNextPage,
    isLoadingMore,
    layoutOptions,
    selectedMediaItem,
    loadMore,
    onSelectedMediaItem,
}: SidebarItemsProps) => {
    const selectedIndex = mediaItems.findIndex((item) => item.id === selectedMediaItem.id);

    const handleSelectionChange = (newKeys: Selection) => {
        const updatedSelectedKeys = new Set(newKeys);
        const firstKey = updatedSelectedKeys.values().next().value;
        const mediaItem = mediaItems.find((item) => item.id === firstKey);

        isNonEmptyString(mediaItem?.id) && onSelectedMediaItem(mediaItem.id);
    };

    const handleDeletedItem = (deletedIds: string[]) => {
        if (deletedIds.includes(String(selectedMediaItem.id))) {
            const nextIndex = selectedIndex + 1;
            const newSelectedIndex = nextIndex < mediaItems.length - 1 ? nextIndex : selectedIndex - 1;
            const newSelectedItem = mediaItems[newSelectedIndex];

            onSelectedMediaItem(newSelectedItem?.id ?? null);
        }
    };

    return (
        <View width={'100%'} height={'100%'}>
            <VirtualizerGridLayout
                items={mediaItems}
                ariaLabel='sidebar-items'
                selectionMode='single'
                selectedKeys={new Set([String(selectedMediaItem.id)])}
                layoutOptions={layoutOptions}
                scrollToIndex={selectedIndex}
                onSelectionChange={handleSelectionChange}
                isLoadingMore={isLoadingMore}
                onLoadMore={() => hasNextPage && loadMore()}
                contentItem={(mediaItem) => (
                    <GridMediaItem
                        contentElement={() => (
                            <MediaThumbnail
                                alt={mediaItem.filename}
                                url={getThumbnailUrl(mediaItem)}
                                onClick={() => onSelectedMediaItem(mediaItem.id ?? null)}
                            />
                        )}
                        topRightElement={() => (
                            <DeleteMediaItem itemsIds={[String(mediaItem.id)]} onDeleted={handleDeletedItem} />
                        )}
                        bottomLeftElement={() => <DownloadDatasetItem mediaItem={mediaItem} />}
                    />
                )}
            />
        </View>
    );
};
