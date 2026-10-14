// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { CSSProperties, Suspense } from 'react';

import { PipelineLink } from '@anomalib-studio/icons';
import {
    Button,
    Content,
    Dialog,
    DialogTrigger,
    dimensionValue,
    Item,
    Loading,
    TabList,
    TabPanels,
    Tabs,
    Text,
    View,
} from '@geti-ui/ui';

import { ModelsList } from './models-list/models-list.component';
import { SinkActions } from './sinks/sink-actions.component';
import { SourceActions } from './sources/source-actions.component';

const paddingStyle = {
    '--spectrum-dialog-padding-x': dimensionValue('size-300'),
    '--spectrum-dialog-padding-y': dimensionValue('size-300'),
} as CSSProperties;

export const PipelineConfiguration = () => {
    return (
        <DialogTrigger type='popover'>
            <Button variant={'secondary'} UNSAFE_style={{ gap: dimensionValue('size-125') }}>
                <PipelineLink fill='white' />
                <Text width={'auto'}>Pipeline configuration</Text>
            </Button>
            <Dialog minWidth={'size-6000'} UNSAFE_style={paddingStyle}>
                <Content>
                    <Tabs aria-label='Dataset import tabs' height={'100%'}>
                        <TabList>
                            <Item key='sources' textValue='Sources'>
                                <Text>Input</Text>
                            </Item>
                            <Item key='model' textValue='Model'>
                                <Text>Model</Text>
                            </Item>
                            <Item key='sinks' textValue='Sinks'>
                                <Text>Output</Text>
                            </Item>
                        </TabList>
                        <TabPanels>
                            <Item key='sources'>
                                <View marginTop={'size-200'}>
                                    <SourceActions />
                                </View>
                            </Item>
                            <Item key='model'>
                                <View marginTop={'size-200'}>
                                    <Suspense fallback={<Loading size='S' />}>
                                        <ModelsList />
                                    </Suspense>
                                </View>
                            </Item>
                            <Item key='sinks'>
                                <View marginTop={'size-200'}>
                                    <SinkActions />
                                </View>
                            </Item>
                        </TabPanels>
                    </Tabs>
                </Content>
            </Dialog>
        </DialogTrigger>
    );
};
