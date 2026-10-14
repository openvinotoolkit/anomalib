import { Button, Flex, Text } from '@geti-ui/ui';
import { Add as AddIcon } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { isEqual } from 'lodash-es';

import { LoadMoreList } from '../../../../../components/load-more-list/load-more-list.component';
import { StatusTag } from '../../../../../components/status-tag/status-tag.component';
import { usePipeline } from '../../../../../hooks/use-pipeline.hook';
import { removeUnderscore } from '../../../utils';
import { SinkConfig } from '../utils';
import { SettingsList } from './settings-list/settings-list.component';
import { SinkIcon } from './sink-icon/sink-icon.component';
import { SinkMenu } from './sink-menu/sink-menu.component';

import classes from './sink-list.module.scss';

type SinksListProps = {
    sinks: SinkConfig[];
    isLoading: boolean;
    hasNextPage: boolean;
    onLoadMore: () => void;
    onAddSink: () => void;
    onEditSink: (config: SinkConfig) => void;
};

type SinksListItemProps = {
    sink: SinkConfig;
    isConnected: boolean;
    onEditSink: (config: SinkConfig) => void;
};

const SinkListItem = ({ sink, isConnected, onEditSink }: SinksListItemProps) => {
    return (
        <Flex
            key={sink.id}
            gap='size-200'
            direction='column'
            UNSAFE_className={clsx(classes.card, {
                [classes.activeCard]: isConnected,
            })}
        >
            <Flex alignItems={'center'} gap={'size-200'}>
                <SinkIcon type={sink.sink_type} />

                <Flex direction={'column'} gap={'size-100'}>
                    <Text UNSAFE_className={classes.title}>{sink.name}</Text>
                    <Flex gap={'size-100'} alignItems={'center'}>
                        <Text UNSAFE_className={classes.type}>{removeUnderscore(sink.sink_type)}</Text>
                        <StatusTag isConnected={isConnected} />
                    </Flex>
                </Flex>
            </Flex>

            <Flex justifyContent={'space-between'}>
                <SettingsList sink={sink} />

                <SinkMenu
                    id={String(sink.id)}
                    name={sink.name}
                    isConnected={isConnected}
                    onEdit={() => onEditSink(sink)}
                />
            </Flex>
        </Flex>
    );
};

export const SinkList = ({ sinks, isLoading, hasNextPage, onLoadMore, onAddSink, onEditSink }: SinksListProps) => {
    const pipeline = usePipeline();
    const currentSinkId = pipeline.data.sink?.id;

    return (
        <LoadMoreList isLoading={isLoading} hasNextPage={hasNextPage} onLoadMore={onLoadMore}>
            <Button variant='secondary' height={'size-800'} UNSAFE_className={classes.addSink} onPress={onAddSink}>
                <AddIcon /> Add new sink
            </Button>

            {sinks.map((sink) => (
                <SinkListItem
                    key={sink.id}
                    sink={sink}
                    isConnected={isEqual(currentSinkId, sink.id)}
                    onEditSink={onEditSink}
                />
            ))}
        </LoadMoreList>
    );
};
