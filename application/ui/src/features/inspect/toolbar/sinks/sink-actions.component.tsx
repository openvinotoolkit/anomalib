// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Flex, Loading, Text } from '@geti-ui/ui';
import { Back } from '@geti-ui/ui/icons';
import { isEmpty } from 'lodash-es';

import { EditSinkForm } from './edit-sink-form.component';
import { useGetSinks } from './hooks/use-get-sinks.hooks';
import { SinkList } from './sink-list/sink-list.component';
import { SinkOptions } from './sink-options.component';
import { SinkConfig } from './utils';

export const SinkActions = () => {
    const { sinks, isFetchingNextPage, isLoading, hasNextPage, fetchNextPage } = useGetSinks();
    const [view, setView] = useState<'list' | 'options' | 'edit'>('list');
    const [currentSink, setCurrentSink] = useState<SinkConfig | null>(null);

    const handleShowList = () => {
        setView('list');
    };

    const handleAddSinks = () => {
        setView('options');
    };

    const handleEditSink = (sink: SinkConfig) => {
        setView('edit');
        setCurrentSink(sink);
    };

    if (isLoading) {
        return <Loading mode={'inline'} size='M' />;
    }

    if (view === 'edit' && !isEmpty(currentSink)) {
        return <EditSinkForm config={currentSink} onSaved={handleShowList} onBackToList={handleShowList} />;
    }

    if (view === 'list') {
        return (
            <SinkList
                sinks={sinks}
                hasNextPage={hasNextPage}
                isLoading={isFetchingNextPage}
                onLoadMore={fetchNextPage}
                onAddSink={handleAddSinks}
                onEditSink={handleEditSink}
            />
        );
    }

    return (
        <SinkOptions onSaved={handleShowList} hasHeader>
            <Flex gap={'size-100'} marginBottom={'size-100'} alignItems={'center'} justifyContent={'space-between'}>
                <ActionButton isQuiet onPress={handleShowList}>
                    <Back />
                </ActionButton>

                <Text>Add new sink</Text>
            </Flex>
        </SinkOptions>
    );
};
