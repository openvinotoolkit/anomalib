// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { Folder } from '@anomalib-studio/icons';
import { toast } from '@anomalib-studio/toast';
import {
    ActionButton,
    FileTrigger,
    Flex,
    Item,
    Key,
    Picker,
    ProgressCircle,
    TextField,
    Tooltip,
    TooltipTrigger,
} from '@geti-ui/ui';
import { Copy } from '@geti-ui/ui/icons';

import { VideoFileSourceConfig } from '../util';

import classes from './video-file-fields.module.scss';

type VideoFileFieldsProps = {
    defaultState: VideoFileSourceConfig;
};

const ACCEPTED_VIDEO_TYPES = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];

export const VideoFileFields = ({ defaultState }: VideoFileFieldsProps) => {
    const { projectId } = useProjectIdentifier();
    const [videoPath, setVideoPath] = useState(defaultState.video_path || '');
    const [selectedFilename, setSelectedFilename] = useState<string | null>(null);
    const [isUploading, setIsUploading] = useState(false);

    // Fetch existing videos for the project
    const {
        data: videosData,
        isLoading: isLoadingVideos,
        refetch: refetchVideos,
    } = $api.useQuery('get', '/api/projects/{project_id}/videos', {
        params: { path: { project_id: projectId } },
    });

    const videos = videosData?.videos ?? [];

    const uploadVideoMutation = $api.useMutation('post', '/api/projects/{project_id}/videos');

    // Find the currently selected video based on video_path
    const getSelectedKey = (): string | undefined => {
        if (selectedFilename) return selectedFilename;
        if (!videoPath) return undefined;
        const video = videos.find((v) => v.video_path === videoPath);
        return video?.filename;
    };

    const handleVideoSelection = (key: Key | null) => {
        if (!key) return;
        const video = videos.find((v) => v.filename === String(key));
        if (video) {
            setVideoPath(video.video_path);
            setSelectedFilename(video.filename);
        }
    };

    const handleVideoUpload = async (files: FileList | null) => {
        if (!files || files.length === 0) return;

        const file = files[0];
        setIsUploading(true);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await uploadVideoMutation.mutateAsync({
                params: { path: { project_id: projectId } },
                // @ts-expect-error OpenAPI type doesn't match FormData
                body: formData,
            });

            // Refetch the videos list to include the new video
            await refetchVideos();

            // Update the video path with the server-side path
            setVideoPath(response.video_path);
            setSelectedFilename(response.filename);

            toast({
                title: 'Video uploaded',
                type: 'success',
                message: `Video "${response.filename}" uploaded successfully`,
            });
        } catch {
            toast({
                title: 'Upload failed',
                type: 'error',
                message: 'Failed to upload video. Please try again.',
            });
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState?.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField isHidden label='video_path' name='video_path' value={videoPath} />

            <TextField isRequired width='100%' label='Name' name='name' defaultValue={defaultState.name} />

            <Flex direction='row' gap='size-200' alignItems='end'>
                <Picker
                    flex='1'
                    isRequired
                    label='Video'
                    items={videos}
                    isLoading={isLoadingVideos}
                    aria-label='Video list'
                    selectedKey={getSelectedKey()}
                    onSelectionChange={handleVideoSelection}
                    placeholder={videos.length === 0 ? 'No videos uploaded yet' : 'Select a video'}
                >
                    {(item) => <Item key={item.filename}>{item.filename}</Item>}
                </Picker>

                {videoPath && (
                    <TooltipTrigger delay={300}>
                        <ActionButton
                            UNSAFE_className={classes.iconButton}
                            aria-label='Copy video path'
                            onPress={() => {
                                navigator.clipboard.writeText(videoPath);
                                toast({ title: 'Copied', type: 'success', message: 'Video path copied to clipboard' });
                            }}
                        >
                            <Copy />
                        </ActionButton>
                        <Tooltip>Copy path to clipboard</Tooltip>
                    </TooltipTrigger>
                )}

                <TooltipTrigger delay={300}>
                    <FileTrigger acceptedFileTypes={ACCEPTED_VIDEO_TYPES} onSelect={handleVideoUpload}>
                        <ActionButton
                            UNSAFE_className={classes.folderIcon}
                            isDisabled={isUploading}
                            aria-label='Upload video file'
                        >
                            {isUploading ? (
                                <ProgressCircle size='S' isIndeterminate aria-label='Uploading' />
                            ) : (
                                <Folder />
                            )}
                        </ActionButton>
                    </FileTrigger>
                    <Tooltip>Upload new video</Tooltip>
                </TooltipTrigger>
            </Flex>
        </Flex>
    );
};
