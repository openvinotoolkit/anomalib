// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex, Switch, TextField } from '@geti-ui/ui';

import { IPCameraSourceConfig } from '../util';

type IpCameraFieldsProps = {
    defaultState: IPCameraSourceConfig;
};

export const IpCameraFields = ({ defaultState }: IpCameraFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField isRequired width={'100%'} label='Name' name='name' defaultValue={defaultState.name} />
            <TextField
                isRequired
                width={'100%'}
                label='Stream Url:'
                name='stream_url'
                defaultValue={defaultState.stream_url}
            />
            <Switch
                name='auth_required'
                aria-label='Require Authentication'
                defaultSelected={defaultState.auth_required}
                key={defaultState.auth_required ? 'true' : 'false'}
            >
                Require Authentication
            </Switch>
        </Flex>
    );
};
