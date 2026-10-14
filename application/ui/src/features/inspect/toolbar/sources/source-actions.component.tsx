// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { ActionButton, Flex, Loading, Text } from '@geti-ui/ui';
import { Back } from '@geti-ui/ui/icons';
import { isEmpty } from 'lodash-es';

import { useGetSources } from '../sinks/hooks/use-get-sources.hooks';
import { EditSourceForm } from './edit-source-form.component';
import { SourcesList } from './source-list/source-list.component';
import { SourceOptions } from './source-options.component';
import { SourceConfig } from './util';

export const SourceActions = () => {
    const [view, setView] = useState<'list' | 'options' | 'edit'>('list');
    const [currentSource, setCurrentSource] = useState<SourceConfig | null>(null);
    const { sources, isFetchingNextPage, isLoading, hasNextPage, fetchNextPage } = useGetSources();

    const handleShowList = () => {
        setView('list');
    };

    const handleAddSource = () => {
        setView('options');
    };

    const handleEditSource = (source: SourceConfig) => {
        setView('edit');
        setCurrentSource(source);
    };

    if (isLoading) {
        return <Loading mode={'inline'} size='M' />;
    }

    if (view === 'edit' && !isEmpty(currentSource)) {
        return <EditSourceForm config={currentSource} onSaved={handleShowList} onBackToList={handleShowList} />;
    }

    if (view === 'list') {
        return (
            <SourcesList
                sources={sources}
                isLoading={isFetchingNextPage}
                hasNextPage={hasNextPage}
                onLoadMore={fetchNextPage}
                onAddSource={handleAddSource}
                onEditSource={handleEditSource}
            />
        );
    }

    return (
        <SourceOptions onSaved={handleShowList} hasHeader={sources.length > 0}>
            <Flex gap={'size-100'} marginBottom={'size-100'} alignItems={'center'} justifyContent={'space-between'}>
                <ActionButton isQuiet onPress={handleShowList}>
                    <Back />
                </ActionButton>

                <Text>Add new input source</Text>
            </Flex>
        </SourceOptions>
    );
};
