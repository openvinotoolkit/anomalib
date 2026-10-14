// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode } from 'react';

import { View } from '@geti-ui/ui';
import { clsx } from 'clsx';
import { isFunction } from 'lodash-es';

import classes from './grid-media-item.module.scss';

interface GridMediaItemProps {
    className?: string;
    contentElement: () => ReactNode;
    topLeftElement?: () => ReactNode;
    topRightElement?: () => ReactNode;
    bottomLeftElement?: () => ReactNode;
    bottomRightElement?: () => ReactNode;
}

export const GridMediaItem = ({
    className,
    contentElement,
    topLeftElement,
    topRightElement,
    bottomLeftElement,
    bottomRightElement,
}: GridMediaItemProps) => {
    return (
        <View width={'100%'} height={'100%'} overflow={'hidden'} UNSAFE_className={clsx(classes.root, className)}>
            {contentElement()}

            {isFunction(topLeftElement) && (
                <View
                    data-floating-container
                    UNSAFE_className={clsx(classes.leftTopElement, classes.floatingContainer)}
                >
                    {topLeftElement()}
                </View>
            )}

            {isFunction(topRightElement) && (
                <View
                    data-floating-container
                    UNSAFE_className={clsx(classes.rightTopElement, classes.floatingContainer)}
                >
                    {topRightElement()}
                </View>
            )}

            {isFunction(bottomLeftElement) && (
                <View
                    data-floating-container
                    UNSAFE_className={clsx(classes.bottomLeftElement, classes.floatingContainer)}
                >
                    {bottomLeftElement()}
                </View>
            )}

            {isFunction(bottomRightElement) && (
                <View
                    data-floating-container
                    UNSAFE_className={clsx(classes.bottomRightElement, classes.floatingContainer)}
                >
                    {bottomRightElement()}
                </View>
            )}
        </View>
    );
};
