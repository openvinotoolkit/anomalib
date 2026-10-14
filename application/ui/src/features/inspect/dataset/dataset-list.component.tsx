import { DialogContainer, Flex, Heading, Selection, Size, View } from '@geti-ui/ui';
import { isNil } from 'lodash-es';
import isEmpty from 'lodash-es/isEmpty';
import { MediaThumbnail } from 'src/components/media-thumbnail/media-thumbnail.component';
import { GridMediaItem } from 'src/components/virtualizer-grid-layout/grid-media-item/grid-media-item.component';
import { VirtualizerGridLayout } from 'src/components/virtualizer-grid-layout/virtualizer-grid-layout.component';

import { getThumbnailUrl } from '../utils';
import { DatasetItemPlaceholder } from './dataset-item-placeholder/dataset-item-placeholder.component';
import { getPlaceholderItem, getPlaceholderKeys, isPlaceholderItem } from './dataset-item-placeholder/util';
import { DeleteMediaItem } from './delete-dataset-item/delete-dataset-item.component';
import { DownloadDatasetItem } from './download-dataset-item/download-dataset-item.component';
import { useGetMediaItems } from './hooks/use-get-media-items.hook';
import { useSelectedMediaItem } from './hooks/use-selected-media-item.hook';
import { MediaPreview } from './media-preview/media-preview.component';
import { InferenceOpacityProvider } from './media-preview/providers/inference-opacity-provider.component';
import { MediaItem } from './types';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from './utils';

const layoutOptions = {
    maxColumns: 4,
    minSpace: new Size(8, 8),
    minItemSize: new Size(120, 120),
    preserveAspectRatio: true,
};

export const DatasetList = () => {
    const [selectedMediaItem, setSelectedMediaItemId] = useSelectedMediaItem();
    const { mediaItems, isFetchingNextPage, hasNextPage, fetchNextPage } = useGetMediaItems();

    const placeholderCount = Math.max(0, REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING - mediaItems.length);
    const placeholderKeys = getPlaceholderKeys(placeholderCount);

    const mediaItemsToRender = [
        ...mediaItems,
        ...Array.from({ length: placeholderCount }).map((_, index): MediaItem => getPlaceholderItem(index)),
    ];

    const handleSelectionChange = (newKeys: Selection) => {
        const updatedSelectedKeys = new Set(newKeys);
        const firstKey = updatedSelectedKeys.values().next().value;
        const itemId = String(firstKey);

        if (!isNil(firstKey) && !isPlaceholderItem(itemId)) {
            setSelectedMediaItemId(itemId);
        }
    };

    return (
        <Flex gap='size-200' direction={'column'} height={'100%'}>
            <Heading>Normal images</Heading>

            <View width={'100%'} height={'100%'}>
                <VirtualizerGridLayout
                    items={mediaItemsToRender}
                    ariaLabel='sidebar-items'
                    selectionMode='single'
                    disabledKeys={placeholderKeys}
                    layoutOptions={layoutOptions}
                    isLoadingMore={isFetchingNextPage}
                    onLoadMore={() => hasNextPage && fetchNextPage()}
                    onSelectionChange={handleSelectionChange}
                    contentItem={(mediaItem) =>
                        mediaItem.filename === 'placeholder' ? (
                            <DatasetItemPlaceholder />
                        ) : (
                            <GridMediaItem
                                contentElement={() => (
                                    <MediaThumbnail
                                        alt={mediaItem.filename}
                                        url={getThumbnailUrl(mediaItem)}
                                        onClick={() => setSelectedMediaItemId(mediaItem.id ?? null)}
                                    />
                                )}
                                topRightElement={() => (
                                    <DeleteMediaItem
                                        itemsIds={[String(mediaItem.id)]}
                                        onDeleted={() => setSelectedMediaItemId(null)}
                                    />
                                )}
                                bottomLeftElement={() => <DownloadDatasetItem mediaItem={mediaItem} />}
                            />
                        )
                    }
                />
            </View>

            <DialogContainer onDismiss={() => setSelectedMediaItemId(null)}>
                {!isEmpty(selectedMediaItem) && (
                    <InferenceOpacityProvider>
                        <MediaPreview
                            mediaItems={mediaItems}
                            loadMore={fetchNextPage}
                            hasNextPage={hasNextPage}
                            isLoadingMore={isFetchingNextPage}
                            selectedMediaItem={selectedMediaItem}
                            onClose={() => setSelectedMediaItemId(null)}
                            onSelectedMediaItem={setSelectedMediaItemId}
                        />
                    </InferenceOpacityProvider>
                )}
            </DialogContainer>
        </Flex>
    );
};
