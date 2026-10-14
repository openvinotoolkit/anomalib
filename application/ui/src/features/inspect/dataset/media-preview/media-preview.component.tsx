// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import {
    Button,
    ButtonGroup,
    Content,
    Dialog,
    dimensionValue,
    Divider,
    Grid,
    Header,
    Heading,
    Size,
    View,
} from '@geti-ui/ui';
import { isEmpty } from 'lodash-es';

import { MediaItem } from '../types';
import { useMediaItemInference } from './hooks/use-media-item-inference.hook';
import { InferenceOpacity } from './inference-opacity/inference-opacity.component';
import { InferenceResult } from './inference-result/inference-result.component';
import { SidebarItems } from './sidebar-items/sidebar-items.component';

type MediaPreviewProps = {
    mediaItems: MediaItem[];
    selectedMediaItem: MediaItem;
    hasNextPage: boolean;
    isLoadingMore: boolean;
    onClose: () => void;
    loadMore: () => void;
    onSelectedMediaItem: (mediaItem: string | null) => Promise<URLSearchParams>;
};

const layoutOptions = {
    maxColumns: 1,
    minSpace: new Size(8, 8),
    minItemSize: new Size(120, 120),
    maxItemSize: new Size(120, 120),
    preserveAspectRatio: true,
};

export const MediaPreview = ({
    mediaItems,
    hasNextPage,
    isLoadingMore,
    selectedMediaItem,
    loadMore,
    onClose,
    onSelectedMediaItem,
}: MediaPreviewProps) => {
    const { data: inferenceResult } = useMediaItemInference(selectedMediaItem);

    return (
        <Dialog UNSAFE_style={{ width: '95vw', height: '95vh' }}>
            <Heading>Preview</Heading>
            <Header>
                <InferenceOpacity isDisabled={isEmpty(inferenceResult?.anomaly_map)} />
            </Header>

            <Divider />

            <Content UNSAFE_style={{ backgroundColor: 'var(--spectrum-global-color-gray-50)' }}>
                <Grid
                    gap='size-125'
                    rows='auto'
                    width='100%'
                    height='100%'
                    columns='1fr 140px'
                    UNSAFE_style={{ padding: dimensionValue('size-125') }}
                    areas={['canvas sidebar', 'canvas sidebar']}
                >
                    <View position={'relative'} gridArea={'canvas'} overflow={'hidden'}>
                        <InferenceResult selectedMediaItem={selectedMediaItem} inferenceResult={inferenceResult} />
                    </View>

                    <View gridArea={'sidebar'}>
                        <SidebarItems
                            mediaItems={mediaItems}
                            hasNextPage={hasNextPage}
                            isLoadingMore={isLoadingMore}
                            layoutOptions={layoutOptions}
                            selectedMediaItem={selectedMediaItem}
                            loadMore={loadMore}
                            onSelectedMediaItem={onSelectedMediaItem}
                        />
                    </View>
                </Grid>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={onClose}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};
