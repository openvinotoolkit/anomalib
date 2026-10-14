// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Key, useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { Heading, InlineAlert, Item, Link, Picker, Text } from '@geti-ui/ui';
import type { SchemaDeviceInfo } from 'src/api/openapi-spec';

import { getDeviceDescription, getDeviceKey, getDeviceLabel, selectPreferredDevice } from './utils/device-metadata';

interface UseTrainingDeviceResult {
    selectedDevice: string | null;
    setSelectedDevice: (device: string | null) => void;
    devices: SchemaDeviceInfo[];
}

export const useTrainingDevice = (): UseTrainingDeviceResult => {
    const { data: availableDevices } = $api.useSuspenseQuery('get', '/api/system/devices/training');
    const devices = availableDevices ?? [];

    const [selectedDevice, setSelectedDevice] = useState<string | null>(() => {
        if (devices.length === 0) {
            return null;
        }
        return selectPreferredDevice(devices) ?? getDeviceKey(devices[0]);
    });

    return {
        selectedDevice,
        setSelectedDevice,
        devices,
    };
};

interface TrainingDevicePickerProps {
    selectedDevice: string | null;
    onDeviceChange: (device: string | null) => void;
    devices: SchemaDeviceInfo[];
}

export const TrainingDevicePicker = ({ selectedDevice, onDeviceChange, devices }: TrainingDevicePickerProps) => {
    const handleDeviceChange = (key: Key | null) => {
        onDeviceChange(key === null ? null : String(key));
    };

    if (devices.length === 0) {
        return (
            <InlineAlert variant='notice'>
                <Heading level={5}>No training devices detected</Heading>
                <Text>
                    Anomalib Studio was unable to discover any compatible training hardware. If you believe this is an
                    error,{' '}
                    <Link
                        isQuiet
                        href='https://github.com/open-edge-platform/anomalib/issues'
                        target='_blank'
                        rel='noreferrer noopener'
                    >
                        let us know on GitHub
                    </Link>
                    .
                </Text>
            </InlineAlert>
        );
    }

    return (
        <Picker
            aria-label='Select a training device'
            selectedKey={selectedDevice}
            onSelectionChange={handleDeviceChange}
            width='size-3400'
            items={devices.map((device) => {
                const key = getDeviceKey(device);
                const label = getDeviceLabel(device);
                const description = getDeviceDescription(device.type);
                return { id: key, label, description };
            })}
        >
            {(item) => (
                <Item key={item.id} textValue={item.label}>
                    <Text>{item.label}</Text>
                    {item.description ? <Text slot='description'>{item.description}</Text> : null}
                </Item>
            )}
        </Picker>
    );
};
