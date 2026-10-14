// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import type { SchemaDeviceInfo } from 'src/api/openapi-spec';

import {
    getDeviceDescription,
    getDeviceKey,
    getDeviceLabel,
    isSamePhysicalDevice,
    selectPreferredDevice,
} from './device-metadata';

const makeDevice = (
    type: string,
    name: string,
    index: number | null = null,
    memory: number | null = null
): SchemaDeviceInfo => ({
    type: type as SchemaDeviceInfo['type'],
    name,
    memory,
    index,
    openvino_name: null,
});

describe('getDeviceKey', () => {
    it('returns type for devices without index', () => {
        expect(getDeviceKey(makeDevice('cpu', 'CPU'))).toBe('cpu');
        expect(getDeviceKey(makeDevice('mps', 'MPS'))).toBe('mps');
    });

    it('returns type-index for devices with index', () => {
        expect(getDeviceKey(makeDevice('xpu', 'Intel(R) Graphics', 0))).toBe('xpu-0');
        expect(getDeviceKey(makeDevice('cuda', 'NVIDIA RTX 4090', 1))).toBe('cuda-1');
    });
});

describe('getDeviceLabel', () => {
    it('returns name for devices without index', () => {
        expect(getDeviceLabel(makeDevice('cpu', 'CPU'))).toBe('CPU');
        expect(getDeviceLabel(makeDevice('mps', 'MPS'))).toBe('MPS');
    });

    it('returns name with [index] for indexed devices', () => {
        expect(getDeviceLabel(makeDevice('xpu', 'Intel(R) Graphics', 0))).toBe('Intel(R) Graphics [0]');
        expect(getDeviceLabel(makeDevice('cuda', 'NVIDIA RTX 4090', 1))).toBe('NVIDIA RTX 4090 [1]');
    });
});

describe('getDeviceDescription', () => {
    it('returns description for known device types', () => {
        expect(getDeviceDescription('cpu')).toBe('High compatibility. Can be slow for large models.');
        expect(getDeviceDescription('XPU')).toBe('Intel unified accelerator architecture.');
        expect(getDeviceDescription('CUDA')).toBe('Accelerated training on NVIDIA CUDA-capable GPUs.');
    });

    it('returns undefined for unknown device types', () => {
        expect(getDeviceDescription('unknown')).toBeUndefined();
    });
});

describe('selectPreferredDevice', () => {
    it('returns null for empty device list', () => {
        expect(selectPreferredDevice([])).toBeNull();
    });

    it('prefers XPU over CPU', () => {
        const devices = [makeDevice('cpu', 'CPU'), makeDevice('xpu', 'Intel(R) Graphics', 0)];
        expect(selectPreferredDevice(devices)).toBe('xpu-0');
    });

    it('prefers GPU/CUDA over CPU', () => {
        const devices = [makeDevice('cpu', 'CPU'), makeDevice('cuda', 'NVIDIA RTX 4090', 0)];
        expect(selectPreferredDevice(devices)).toBe('cuda-0');
    });

    it('returns first device key if no preferred devices found', () => {
        // All known types are in DEVICE_PRIORITY, so this tests an unlikely edge case
        const devices = [makeDevice('cpu', 'CPU')];
        expect(selectPreferredDevice(devices)).toBe('cpu');
    });

    it('respects priority order', () => {
        // XPU has highest priority
        const devices1 = [
            makeDevice('cpu', 'CPU'),
            makeDevice('cuda', 'NVIDIA GPU', 0),
            makeDevice('xpu', 'Intel(R) Graphics', 0),
        ];
        expect(selectPreferredDevice(devices1)).toBe('xpu-0');

        // CUDA before MPS
        const devices2 = [makeDevice('cpu', 'CPU'), makeDevice('mps', 'MPS'), makeDevice('cuda', 'NVIDIA GPU', 0)];
        expect(selectPreferredDevice(devices2)).toBe('cuda-0');

        // MPS before CPU
        const devices3 = [makeDevice('cpu', 'CPU'), makeDevice('mps', 'MPS')];
        expect(selectPreferredDevice(devices3)).toBe('mps');
    });

    it('handles multiple devices of the same type', () => {
        const devices = [
            makeDevice('cpu', 'CPU'),
            makeDevice('xpu', 'Intel(R) Graphics', 0),
            makeDevice('xpu', 'Intel(R) Graphics', 1),
        ];
        // Should prefer the first XPU found
        expect(selectPreferredDevice(devices)).toBe('xpu-0');
    });
});

describe('isSamePhysicalDevice', () => {
    it('returns false when there is no selected training device', () => {
        expect(isSamePhysicalDevice(undefined, 'CPU')).toBe(false);
    });

    it('returns false when there is no inference device', () => {
        expect(isSamePhysicalDevice(makeDevice('cpu', 'CPU'), null)).toBe(false);
        expect(isSamePhysicalDevice(makeDevice('cpu', 'CPU'), undefined)).toBe(false);
        expect(isSamePhysicalDevice(makeDevice('cpu', 'CPU'), '')).toBe(false);
    });

    it('matches CPU training device with CPU inference device', () => {
        expect(isSamePhysicalDevice(makeDevice('cpu', 'CPU'), 'CPU')).toBe(true);
    });

    it('matches an Intel GPU (xpu) training device with the same-index GPU inference device', () => {
        expect(isSamePhysicalDevice(makeDevice('xpu', 'Intel(R) Graphics', 0), 'GPU.0')).toBe(true);
    });

    it('does not match an Intel GPU (xpu) training device with a different-index GPU inference device', () => {
        expect(isSamePhysicalDevice(makeDevice('xpu', 'Intel(R) Graphics', 0), 'GPU.1')).toBe(false);
        expect(isSamePhysicalDevice(makeDevice('xpu', 'Intel(R) Graphics', 1), 'GPU.0')).toBe(false);
    });

    it('is case-insensitive when comparing device types', () => {
        expect(isSamePhysicalDevice(makeDevice('xpu', 'Intel(R) Graphics', 0), 'gpu.0')).toBe(true);
    });

    it('never conflicts for CUDA training devices (unsupported by OpenVINO inference)', () => {
        expect(isSamePhysicalDevice(makeDevice('cuda', 'NVIDIA RTX 4090', 0), 'GPU.0')).toBe(false);
        expect(isSamePhysicalDevice(makeDevice('cuda', 'NVIDIA RTX 4090', 0), 'CPU')).toBe(false);
    });

    it('never conflicts for MPS training devices (unsupported by OpenVINO inference)', () => {
        expect(isSamePhysicalDevice(makeDevice('mps', 'MPS'), 'CPU')).toBe(false);
    });

    it('does not match a CPU training device with a GPU inference device', () => {
        expect(isSamePhysicalDevice(makeDevice('cpu', 'CPU'), 'GPU.0')).toBe(false);
    });
});
