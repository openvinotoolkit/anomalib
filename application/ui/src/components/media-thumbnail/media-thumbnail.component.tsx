// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Skeleton } from '@geti-ui/ui';

import classes from './media-thumbnail.module.scss';

type MediaThumbnailProps = {
    onClick?: () => void;
    onDoubleClick?: () => void;
    url: string;
    alt: string;
};

const RETRY_LIMIT = 3;

export const MediaThumbnail = ({ onDoubleClick, onClick, url, alt }: MediaThumbnailProps) => {
    const [retry, setRetry] = useState(0);
    const [isLoading, setIsLoading] = useState(true);

    const handleError = () => {
        if (retry < RETRY_LIMIT) {
            setRetry((current) => current + 1);
            setIsLoading(true);
        } else {
            setIsLoading(false);
        }
    };

    const handleLoad = () => {
        setIsLoading(false);
    };

    return (
        <div onDoubleClick={onDoubleClick} onClick={onClick} style={{ height: '100%' }}>
            {isLoading && <Skeleton width={'100%'} height={'100%'} className={classes.loader} />}

            <img src={`${url}?retry=${retry}`} alt={alt} onLoad={handleLoad} onError={handleError} />
        </div>
    );
};
