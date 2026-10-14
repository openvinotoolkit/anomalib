// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, dimensionValue, Flex, Loading, Text, View } from '@geti-ui/ui';
import { Refresh } from '@geti-ui/ui/icons';
import { useActivateAndRunPipeline, usePipeline } from 'src/hooks/use-pipeline.hook';

import { useStreamConnection } from '../../../components/stream/stream-connection-provider';
import { PlayStreamButton } from './play-stream-button/play-stream-button.component';
import { Stream } from './stream/stream.component';

const RECONNECT_CLEANUP_DELAY_MS = 300; // Delay to allow stream connection cleanup to complete before reconnecting

export const StreamContainer = ({ hasActiveProject }: { hasActiveProject: boolean }) => {
    const { data: pipeline } = usePipeline();
    const { start, stop, status } = useStreamConnection();
    const activateAndRunPipeline = useActivateAndRunPipeline({ onSuccess: start });

    const handleReconnect = async () => {
        try {
            // Stop the old connection first to clean it up
            await stop();
            // Wait for cleanup to complete and status to update to 'idle'
            await new Promise((resolve) => setTimeout(resolve, RECONNECT_CLEANUP_DELAY_MS));

            // If pipeline is already running, just start the stream directly
            // Otherwise, activate the pipeline which will start streaming via onSuccess callback
            if (pipeline?.status === 'running') {
                await start();
            } else {
                await activateAndRunPipeline.mutateAsync();
            }
        } catch (error) {
            console.error('Failed to reconnect stream:', error);
        }
    };

    if (!hasActiveProject) {
        return <PlayStreamButton isDisabled />;
    }

    if (activateAndRunPipeline.isPending) {
        return (
            <View backgroundColor={'gray-200'} width='90%' height='90%'>
                <Flex alignItems={'center'} justifyContent={'center'} height='100%'>
                    <Loading mode='inline' />
                </Flex>
            </View>
        );
    }

    return (
        <>
            {status === 'idle' && <PlayStreamButton onStart={start} isDisabled={activateAndRunPipeline.isPending} />}

            {(status === 'disconnected' || status === 'failed') && (
                <View backgroundColor={'gray-200'} width='90%' height='90%'>
                    <Flex
                        alignItems={'center'}
                        justifyContent={'center'}
                        height='100%'
                        direction='column'
                        gap='size-200'
                    >
                        <Text>Stream disconnected</Text>
                        <Button
                            onPress={handleReconnect}
                            aria-label={'Reconnect stream'}
                            isDisabled={activateAndRunPipeline.isPending}
                            variant='primary'
                        >
                            <Refresh style={{ marginInlineEnd: dimensionValue('size-100') }} />
                            Reconnect
                        </Button>
                    </Flex>
                </View>
            )}

            {(status === 'connecting' || status === 'connected') && <Stream />}
        </>
    );
};
