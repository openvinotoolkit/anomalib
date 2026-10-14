import { Button, Flex, Text } from '@geti-ui/ui';
import { Add as AddIcon } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { isEqual } from 'lodash-es';

import { LoadMoreList } from '../../../../../components/load-more-list/load-more-list.component';
import { StatusTag } from '../../../../../components/status-tag/status-tag.component';
import { usePipeline } from '../../../../../hooks/use-pipeline.hook';
import { removeUnderscore } from '../../../utils';
import { SourceMenu } from '../source-menu/source-menu.component';
import { SourceConfig } from '../util';
import { SettingsList } from './settings-list/settings-list.component';
import { SourceIcon } from './source-icon/source-icon.component';

import classes from './source-list.module.scss';

type SourcesListProps = {
    sources: SourceConfig[];
    isLoading: boolean;
    hasNextPage: boolean;
    onLoadMore: () => void;
    onAddSource: () => void;
    onEditSource: (config: SourceConfig) => void;
};

type SourceListItemProps = {
    source: SourceConfig;
    isConnected: boolean;
    onEditSource: (config: SourceConfig) => void;
};

const SourceListItem = ({ source, isConnected, onEditSource }: SourceListItemProps) => {
    return (
        <Flex
            key={source.id}
            gap='size-200'
            direction='column'
            UNSAFE_className={clsx(classes.card, { [classes.activeCard]: isConnected })}
        >
            <Flex alignItems={'center'} gap={'size-200'}>
                <SourceIcon type={source.source_type} />

                <Flex direction={'column'} gap={'size-100'}>
                    <Text UNSAFE_className={classes.title}>{source.name}</Text>
                    <Flex gap={'size-100'} alignItems={'center'}>
                        <Text UNSAFE_className={classes.type}>{removeUnderscore(source.source_type)}</Text>
                        <StatusTag isConnected={isConnected} />
                    </Flex>
                </Flex>
            </Flex>

            <Flex justifyContent={'space-between'}>
                <SettingsList source={source} />

                <SourceMenu
                    id={String(source.id)}
                    name={source.name}
                    isConnected={isConnected}
                    onEdit={() => onEditSource(source)}
                />
            </Flex>
        </Flex>
    );
};

export const SourcesList = ({
    sources,
    isLoading,
    hasNextPage,
    onLoadMore,
    onAddSource,
    onEditSource,
}: SourcesListProps) => {
    const pipeline = usePipeline();
    const currentSourceId = pipeline.data.source?.id;

    return (
        <LoadMoreList isLoading={isLoading} hasNextPage={hasNextPage} onLoadMore={onLoadMore}>
            <Button variant='secondary' height={'size-800'} UNSAFE_className={classes.addSource} onPress={onAddSource}>
                <AddIcon /> Add new source
            </Button>

            {sources.map((source) => (
                <SourceListItem
                    key={source.id}
                    source={source}
                    isConnected={isEqual(currentSourceId, source.id)}
                    onEditSource={onEditSource}
                />
            ))}
        </LoadMoreList>
    );
};
