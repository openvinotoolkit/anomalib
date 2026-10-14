// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Dispatch, RefObject, SetStateAction, useCallback, useEffect, useRef, useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import { Button, dimensionValue, Flex } from '@geti-ui/ui';
import { clsx } from 'clsx';

import { useStreamConnection } from '../../../../components/stream/stream-connection-provider';
import { ZoomTransform } from '../../../../components/zoom/zoom-transform';
import { useEventListener } from '../../../../hooks/event-listener/event-listener.hook';
import { Fps } from '../fps/fps.component';

import classes from './stream.module.scss';

const useSetTargetSizeBasedOnImage = (
    setSize: Dispatch<SetStateAction<{ width: number; height: number }>>,
    imageRef: RefObject<HTMLImageElement | null>,
    streamUrl: string | null
) => {
    useEffect(() => {
        const image = imageRef.current;
        if (!image) return;

        const updateSize = () => {
            if (image.naturalWidth && image.naturalHeight) {
                setSize({ width: image.naturalWidth, height: image.naturalHeight });
            }
        };

        const resizeObserver = new ResizeObserver(updateSize);
        image.addEventListener('load', updateSize);
        resizeObserver.observe(image);

        return () => {
            image.removeEventListener('load', updateSize);
            resizeObserver.disconnect();
        };
    }, [setSize, imageRef, streamUrl]);

    useEffect(() => {
        const image = imageRef.current;
        if (!image || !streamUrl) {
            return;
        }

        let attempts = 0;
        const maxAttempts = 20;
        const interval = setInterval(() => {
            if (image.naturalWidth && image.naturalHeight) {
                setSize({ width: image.naturalWidth, height: image.naturalHeight });
                clearInterval(interval);
                return;
            }

            attempts += 1;
            if (attempts >= maxAttempts) {
                clearInterval(interval);
            }
        }, 200);

        return () => {
            clearInterval(interval);
        };
    }, [setSize, imageRef, streamUrl]);
};

export const Stream = () => {
    const imageRef = useRef<HTMLImageElement>(null);
    const { projectId } = useProjectIdentifier();
    const [size, setSize] = useState({ height: 608, width: 892 });
    const [hasCaptureAnimation, setHasCaptureAnimation] = useState(false);
    const { stop: stopStream, setStatus, status, streamUrl } = useStreamConnection();

    useSetTargetSizeBasedOnImage(setSize, imageRef, streamUrl);
    useEventListener('animationend', () => setHasCaptureAnimation(false), imageRef);

    const handleStreamLoad = () => {
        // Guard against spurious loads (e.g. the `data:,` src swap during stop) that would
        // otherwise resurrect a `connected` state after the stream was torn down.
        if (!streamUrl) {
            return;
        }

        setStatus((current) => (current === 'connecting' ? 'connected' : current));
    };

    const handleStreamError = useCallback(() => {
        // Only set failed if we were previously connected or connecting,
        // and avoid flipping to failed if the url was just cleared (null)
        if (streamUrl) {
            setStatus((current) => (current === 'connected' ? 'disconnected' : 'failed'));
            toast({ type: 'error', message: 'Stream connection failed' });
        }
    }, [setStatus, streamUrl]);

    const captureImageMutation = $api.useMutation('get', '/api/projects/{project_id}/capture', {
        onError: () => {
            toast({ type: 'error', message: `Failed to upload 1 item` });
        },
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/images', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    const handleCaptureFrame = async () => {
        setHasCaptureAnimation(true);

        await captureImageMutation.mutateAsync({
            params: { path: { project_id: projectId } },
        });
    };

    const handleStopStream = async () => {
        // cleanup runs on <img> detach and aborts the MJPEG multipart request,
        // which browsers otherwise keep streaming even after src is cleared or the element unmounts.
        if (imageRef.current) {
            imageRef.current.src = 'data:,';
        }

        await stopStream();
    };

    return (
        <Flex
            width={'100%'}
            height={'100%'}
            position={'relative'}
            direction={'column'}
            alignItems={'center'}
            justifyContent={'center'}
            UNSAFE_style={{ paddingBlockEnd: dimensionValue('size-400') }}
        >
            {status === 'connected' && <Fps projectId={projectId} />}

            <ZoomTransform target={size}>
                <button
                    type='button'
                    aria-label='Pause stream'
                    onClick={handleStopStream}
                    className={classes.pauseStream}
                >
                    <img
                        key={streamUrl}
                        ref={imageRef}
                        src={streamUrl ?? undefined}
                        width={size.width}
                        height={size.height}
                        alt='stream'
                        aria-label='stream player'
                        onLoad={handleStreamLoad}
                        onError={handleStreamError}
                        style={{ background: 'var(--spectrum-global-color-gray-200)' }}
                        className={clsx({ [classes.takeOldCamera]: hasCaptureAnimation })}
                    />
                </button>
            </ZoomTransform>

            {status === 'connected' && (
                <Button onPress={handleCaptureFrame} variant='primary' UNSAFE_className={classes.captureButton}>
                    Capture
                </Button>
            )}
        </Flex>
    );
};
