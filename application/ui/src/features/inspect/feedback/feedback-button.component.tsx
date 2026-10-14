// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { fetchClient } from '@anomalib-studio/api';
import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Heading,
    Item,
    Picker,
    Text,
    TextArea,
    Tooltip,
    TooltipTrigger,
    View,
} from '@geti-ui/ui';
import { CheckmarkCircleOutline, DownloadIcon, ExternalLinkIcon, HelpIcon } from '@geti-ui/ui/icons';
import { useMutation } from '@tanstack/react-query';
import { isTauri } from '@tauri-apps/api/core';

import { downloadBlob } from '../utils';

const GITHUB_ISSUES_URL = 'https://github.com/open-edge-platform/anomalib/issues/new';

interface LibraryVersions {
    python: string;
    pytorch: string | null;
    lightning: string | null;
    torchmetrics: string | null;
    openvino: string | null;
    onnx: string | null;
    anomalib: string | null;
    cuda: string | null;
    cudnn: string | null;
    xpu_driver: string | null;
}

interface DeviceInfo {
    type: string;
    name: string;
    memory: number | null;
    index: number | null;
}

interface SystemInfo {
    os_name: string;
    os_version: string;
    platform: string;
    app_version: string;
    libraries: LibraryVersions;
    devices: DeviceInfo[];
}

type IssueType = 'bug' | 'feature';

const ISSUE_TYPE_OPTIONS = [
    { id: 'bug', label: 'Bug Report' },
    { id: 'feature', label: 'Feature Request' },
] as const;

const fetchSystemInfo = async (): Promise<SystemInfo> => {
    const response = await fetchClient.GET('/api/system/info', {});
    return response.data as SystemInfo;
};

const formatLibraryVersion = (name: string, version: string | null): string => {
    return version ? `${name}: ${version}` : `${name}: not installed`;
};

const formatDeviceInfo = (devices: DeviceInfo[], libraries: LibraryVersions): string[] => {
    const lines: string[] = [];

    for (const device of devices) {
        const memoryStr = device.memory ? ` (${(device.memory / 1024 ** 3).toFixed(1)} GB)` : '';
        const indexStr = device.index !== null ? ` [${device.index}]` : '';
        lines.push(`${device.type}${indexStr}: ${device.name}${memoryStr}`);
    }

    // Add driver/runtime versions from libraries
    if (libraries.cuda) {
        lines.push(`CUDA: ${libraries.cuda}`);
    }
    if (libraries.cudnn) {
        lines.push(`cuDNN: ${libraries.cudnn}`);
    }
    if (libraries.xpu_driver) {
        lines.push(`XPU Driver: ${libraries.xpu_driver}`);
    }

    if (lines.length === 0) {
        lines.push('No devices detected');
    }

    return lines;
};

const sanitizeDescription = (description: string): string => {
    // Remove non-printable control characters except for common whitespace
    const cleaned = description.replace(/[^\x09\x0A\x0D\x20-\x7E\xA0-\uFFFF]/g, '');
    const trimmed = cleaned.trim();
    const MAX_LENGTH = 4000;
    if (trimmed.length <= MAX_LENGTH) {
        return trimmed;
    }
    return `${trimmed.slice(0, MAX_LENGTH)}\n\n[description truncated]`;
};

const createGitHubIssueUrl = (systemInfo: SystemInfo, issueType: IssueType, description: string): string => {
    const isBug = issueType === 'bug';
    const title = isBug ? '[Bug]: ' : '[Feature]: ';
    const labels = isBug ? ['bug', 'Anomalib Studio'] : ['enhancement', 'Anomalib Studio'];
    const sanitizedDescription = sanitizeDescription(description);

    const { libraries, devices } = systemInfo;

    const libraryLines = [
        formatLibraryVersion('Python', libraries.python),
        formatLibraryVersion('PyTorch', libraries.pytorch),
        formatLibraryVersion('Lightning', libraries.lightning),
        formatLibraryVersion('TorchMetrics', libraries.torchmetrics),
        formatLibraryVersion('OpenVINO', libraries.openvino),
        formatLibraryVersion('ONNX', libraries.onnx),
        formatLibraryVersion('Anomalib', libraries.anomalib),
    ];

    const deviceLines = formatDeviceInfo(devices, libraries);

    const body = `## System Information

### Environment
- **OS**: ${systemInfo.os_name} ${systemInfo.os_version}
- **Platform**: ${systemInfo.platform}
- **App**: Anomalib Studio v${systemInfo.app_version}

### Library Versions
${libraryLines.map((line) => `- ${line}`).join('\n')}

### Devices
${deviceLines.map((line) => `- ${line}`).join('\n')}

## ${isBug ? 'Bug Description' : 'Feature Description'}
${sanitizedDescription || '_Please describe the issue or feature request_'}
`;

    const params = new URLSearchParams({
        title,
        body,
        labels: labels.join(','),
    });

    return `${GITHUB_ISSUES_URL}?${params.toString()}`;
};

const downloadLogs = async (): Promise<void> => {
    const response = await fetchClient.POST('/api/system/logs:export', {
        parseAs: 'blob',
    });

    if (response.data) {
        const contentDisposition = response.response.headers.get('content-disposition');
        const filenameMatch = contentDisposition?.match(/filename=(.+)/);
        const filename = filenameMatch ? filenameMatch[1] : 'anomalib_studio_logs.zip';

        downloadBlob(response.data as Blob, filename);
    }
};

interface SubmitFeedbackParams {
    issueType: IssueType;
    description: string;
}

/**
 * Opens the GitHub issue URL in the user's default browser.
 * In Tauri, window.open() is silently blocked by the webview, so we use the
 * shell plugin to launch the URL externally. Falls back to window.open() if
 * the shell plugin is unavailable or in a regular browser environment.
 * @see https://github.com/open-edge-platform/anomalib/issues/3425
 */
const submitFeedback = async ({ issueType, description }: SubmitFeedbackParams): Promise<void> => {
    const systemInfo = await fetchSystemInfo();
    const issueUrl = createGitHubIssueUrl(systemInfo, issueType, description);
    if (isTauri()) {
        try {
            const { open } = await import('@tauri-apps/plugin-shell');
            await open(issueUrl);
        } catch {
            // Fallback to browser API if shell plugin fails
            window.open(issueUrl, '_blank', 'noopener,noreferrer');
        }
    } else {
        window.open(issueUrl, '_blank', 'noopener,noreferrer');
    }
};

interface FeedbackDialogContentProps {
    close: () => void;
}

const FeedbackDialogContent = ({ close }: FeedbackDialogContentProps) => {
    const [issueType, setIssueType] = useState<IssueType>('bug');
    const [description, setDescription] = useState('');

    const submitMutation = useMutation({
        mutationFn: submitFeedback,
        onSuccess: () => close(),
    });

    const downloadLogsMutation = useMutation({
        mutationFn: downloadLogs,
    });

    const error = submitMutation.error || downloadLogsMutation.error;

    return (
        <>
            <Heading>Submit Feedback</Heading>
            <Divider />
            <Content>
                <Flex direction='column' gap='size-200'>
                    <Text>
                        Help us improve by reporting bugs or suggesting new features. Your feedback will be submitted as
                        a GitHub issue with system information automatically included.
                    </Text>

                    <Picker
                        label='Issue Type'
                        selectedKey={issueType}
                        onSelectionChange={(key) => setIssueType(key as IssueType)}
                        width='100%'
                    >
                        {ISSUE_TYPE_OPTIONS.map((option) => (
                            <Item key={option.id}>{option.label}</Item>
                        ))}
                    </Picker>

                    <TextArea
                        label='Description'
                        placeholder={
                            issueType === 'bug'
                                ? 'Describe what happened and what you expected to happen...'
                                : 'Describe the feature you would like to see...'
                        }
                        value={description}
                        onChange={setDescription}
                        width='100%'
                        height='size-1600'
                    />

                    {issueType === 'bug' && (
                        <View backgroundColor='gray-100' padding='size-150' borderRadius='regular'>
                            {downloadLogsMutation.isSuccess ? (
                                <Flex direction='column' gap='size-150'>
                                    <Flex alignItems='center' gap='size-100'>
                                        <CheckmarkCircleOutline size='S' color='positive' />
                                        <Text
                                            UNSAFE_style={{
                                                fontSize: '12px',
                                                color: 'var(--spectrum-global-color-green-700)',
                                            }}
                                        >
                                            Logs downloaded! Remember to attach them to your GitHub issue.
                                        </Text>
                                    </Flex>
                                    <Button
                                        variant='secondary'
                                        isQuiet
                                        onPress={() => downloadLogsMutation.mutate()}
                                        isPending={downloadLogsMutation.isPending}
                                        isDisabled={submitMutation.isPending}
                                    >
                                        <DownloadIcon />
                                        <Text>Download Again</Text>
                                    </Button>
                                </Flex>
                            ) : (
                                <Flex direction='row' alignItems='center' justifyContent='space-between' gap='size-100'>
                                    <Text
                                        UNSAFE_style={{
                                            fontSize: '12px',
                                            color: 'var(--spectrum-global-color-gray-700)',
                                        }}
                                    >
                                        Optionally download and attach application logs to help diagnose the issue.
                                    </Text>
                                    <Button
                                        variant='secondary'
                                        onPress={() => downloadLogsMutation.mutate()}
                                        isPending={downloadLogsMutation.isPending}
                                        isDisabled={submitMutation.isPending}
                                    >
                                        <DownloadIcon />
                                        <Text>Download Logs</Text>
                                    </Button>
                                </Flex>
                            )}
                        </View>
                    )}

                    {error && (
                        <View
                            backgroundColor='negative'
                            padding='size-100'
                            borderRadius='regular'
                            UNSAFE_style={{ color: 'var(--spectrum-global-color-red-700)' }}
                        >
                            <Text>
                                {error instanceof Error ? error.message : 'An error occurred. Please try again.'}
                            </Text>
                        </View>
                    )}
                </Flex>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Cancel
                </Button>
                <Button
                    variant='accent'
                    onPress={() => submitMutation.mutate({ issueType, description })}
                    isPending={submitMutation.isPending}
                >
                    <ExternalLinkIcon />
                    <Text>Open GitHub Issue</Text>
                </Button>
            </ButtonGroup>
        </>
    );
};

export const FeedbackButton = () => {
    return (
        <DialogTrigger type='modal'>
            <TooltipTrigger delay={0}>
                <ActionButton isQuiet aria-label='Submit feedback'>
                    <HelpIcon />
                </ActionButton>
                <Tooltip>Submit Feedback</Tooltip>
            </TooltipTrigger>
            {(close) => (
                <Dialog>
                    <FeedbackDialogContent close={close} />
                </Dialog>
            )}
        </DialogTrigger>
    );
};
