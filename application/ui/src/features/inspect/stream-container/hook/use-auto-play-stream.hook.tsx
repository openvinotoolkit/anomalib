// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef } from 'react';

import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import { usePipeline, useRunPipeline } from 'src/hooks/use-pipeline.hook';

import { useStreamConnection } from '../../../../components/stream/stream-connection-provider';
import { isNonEmptyString } from '../../utils';

export const STREAM_ERROR_MESSAGE = 'Failed to connect to the stream';

const ERROR_TOAST_DELAY = 500; // Delay before showing error toast to allow recovery

export const useAutoPlayStream = () => {
    const runPipeline = useRunPipeline({});
    const { data: pipeline } = usePipeline();
    const { start, status } = useStreamConnection();
    const { projectId } = useProjectIdentifier();

    const hasModel = isNonEmptyString(pipeline?.model?.id);
    const hasSource = isNonEmptyString(pipeline?.source?.id);
    const isActive = pipeline?.status === 'active';
    const hasInferenceConfig = hasModel && hasSource;

    // Track previous status to detect transitions
    const previousStatusRef = useRef<typeof status>('idle');
    // Track current status for timeout callback
    const currentStatusRef = useRef<typeof status>(status);
    // Track timeout for debounced error toast
    const errorToastTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

    useEffect(() => {
        const previousStatus = previousStatusRef.current;
        const isTransitioningToFailed = previousStatus !== 'failed' && status === 'failed';
        const isRecoveringFromFailed = previousStatus === 'failed' && status !== 'failed';

        // Update current status ref
        currentStatusRef.current = status;

        // Clear any pending error toast if status recovers
        if (isRecoveringFromFailed && errorToastTimeoutRef.current) {
            clearTimeout(errorToastTimeoutRef.current);
            errorToastTimeoutRef.current = null;
        }

        // Only show error toast when transitioning TO 'failed' status
        if (isTransitioningToFailed) {
            // Debounce error toast to allow connection to recover
            errorToastTimeoutRef.current = setTimeout(() => {
                // Double-check status is still 'failed' before showing toast
                // Use ref to get the latest status value
                if (currentStatusRef.current === 'failed') {
                    toast({ type: 'error', message: STREAM_ERROR_MESSAGE });
                }
                errorToastTimeoutRef.current = null;
            }, ERROR_TOAST_DELAY);
        }

        // Update previous status
        previousStatusRef.current = status;

        if (hasSource && status === 'idle') {
            start();
        }

        if (hasInferenceConfig && isActive && !runPipeline.isPending) {
            runPipeline.mutate({ params: { path: { project_id: projectId } } });
        }

        // Cleanup timeout on unmount
        return () => {
            if (errorToastTimeoutRef.current) {
                clearTimeout(errorToastTimeoutRef.current);
                errorToastTimeoutRef.current = null;
            }
        };
    }, [hasSource, status, start, hasInferenceConfig, isActive, runPipeline, projectId]);
};
