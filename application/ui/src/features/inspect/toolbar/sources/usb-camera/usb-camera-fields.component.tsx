// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useRef, useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { ActionButton, Flex, Item, Key, Loading, Picker, TextField } from '@geti-ui/ui';
import { Refresh } from '@geti-ui/ui/icons';
import { isEmpty } from 'lodash-es';

import { UsbCameraSourceConfig } from '../util';

type UsbCameraFieldsProps = {
    defaultState: UsbCameraSourceConfig;
};

const findDeviceName = (devices: { id: number; name: string }[], deviceId: number): string | undefined =>
    devices.find(({ id }) => id === deviceId)?.name;

export const UsbCameraFields = ({ defaultState }: UsbCameraFieldsProps) => {
    const [userOverrideName, setUserOverrideName] = useState<string | null>(
        isEmpty(defaultState.name) ? null : defaultState.name
    );

    const isSystemName = useRef(isEmpty(defaultState.name));

    const {
        data: cameraDevices,
        isLoading,
        isRefetching,
        refetch,
    } = $api.useQuery('get', '/api/system/devices/camera');

    const devices = (cameraDevices ?? []).map((device) => ({ id: device.index, name: device.name }));
    const name = userOverrideName ?? findDeviceName(devices, defaultState.device_id) ?? '';

    const handleNameChange = (value: string) => {
        setUserOverrideName(value);
        isSystemName.current = false;
    };

    const handleSelectionChange = (key: Key | null) => {
        if (key === null) {
            return;
        }

        const deviceName = findDeviceName(devices, Number(key));

        if (deviceName && isSystemName.current) {
            setUserOverrideName(deviceName);
        }
    };

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField isHidden label='name' name='name' value={name} />
            <TextField
                isRequired
                width='100%'
                label='Name'
                name='name_display'
                value={name}
                onChange={handleNameChange}
            />

            <Flex alignItems='end' gap='size-200'>
                <Picker
                    flex='1'
                    isRequired
                    label='Camera'
                    name='device_id'
                    items={devices}
                    isLoading={isLoading}
                    aria-label='Camera list'
                    defaultSelectedKey={String(defaultState.device_id)}
                    onSelectionChange={handleSelectionChange}
                >
                    {(item) => <Item key={item.id}>{item.name}</Item>}
                </Picker>

                <ActionButton
                    isQuiet
                    onPress={() => refetch()}
                    aria-label='Refresh Cameras'
                    isDisabled={isLoading || isRefetching}
                >
                    {isRefetching ? <Loading mode={'inline'} size='S' /> : <Refresh />}
                </ActionButton>
            </Flex>
        </Flex>
    );
};
