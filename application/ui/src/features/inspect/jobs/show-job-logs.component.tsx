// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Suspense, useMemo } from 'react';

import {
    ActionButton,
    Button,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Heading,
    Icon,
    Loading,
} from '@geti-ui/ui';
import { LogsIcon } from '@geti-ui/ui/icons';
import { queryOptions, experimental_streamedQuery as streamedQuery, useQuery } from '@tanstack/react-query';
import { fetchSSE } from 'src/api/fetch-sse';

import { LogEntry } from './log-types';
import { LogViewer } from './log-viewer.component';

const JobLogsDialogContent = ({ jobId }: { jobId: string }) => {
    const query = useQuery(
        queryOptions({
            queryKey: ['get', '/api/jobs/{job_id}/logs', jobId],
            queryFn: streamedQuery<LogEntry>({
                streamFn: () => fetchSSE<LogEntry>(`/api/jobs/${jobId}/logs`),
            }),
            staleTime: Infinity,
        })
    );

    // Filter out any malformed log entries and ensure we have valid LogEntry objects
    const validLogs = useMemo(() => {
        const entries = query.data ?? [];
        return entries.filter((entry): entry is LogEntry => {
            return (
                entry !== null &&
                typeof entry === 'object' &&
                'record' in entry &&
                entry.record !== null &&
                typeof entry.record === 'object' &&
                'level' in entry.record &&
                'time' in entry.record &&
                'message' in entry.record
            );
        });
    }, [query.data]);

    return <LogViewer logs={validLogs} isLoading={query.isLoading} />;
};

export const JobLogsDialog = ({ close, jobId }: { close: () => void; jobId: string }) => {
    return (
        <Dialog>
            <Heading>Job Logs</Heading>
            <Divider />
            <Content>
                <Suspense fallback={<Loading mode='inline' />}>
                    <JobLogsDialogContent jobId={jobId} />
                </Suspense>
            </Content>
            <ButtonGroup>
                <Button variant='secondary' onPress={close}>
                    Close
                </Button>
            </ButtonGroup>
        </Dialog>
    );
};

export const ShowJobLogs = ({ jobId }: { jobId: string }) => {
    return (
        <DialogTrigger type='fullscreen'>
            <ActionButton aria-label='View job logs'>
                <Icon>
                    <LogsIcon />
                </Icon>
            </ActionButton>
            {(close) => <JobLogsDialog close={close} jobId={jobId} />}
        </DialogTrigger>
    );
};
