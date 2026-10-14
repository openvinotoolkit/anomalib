// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { DependencyList, RefObject, useEffect } from 'react';

import { isNil } from 'lodash-es';

interface useGetTargetPositionProps {
    gap: number;
    ref: RefObject<HTMLDivElement | null>;
    delay?: number;
    scrollToIndex?: number;
    dependencies?: DependencyList;
    callback: (scrollTo: number) => void;
}

const isValidIndex = (index?: number): index is number => !isNil(index) && Number.isInteger(index) && index >= 0;

export const useGetTargetPosition = ({ gap, delay = 500, ref, scrollToIndex, callback }: useGetTargetPositionProps) => {
    useEffect(() => {
        if (isNil(scrollToIndex)) {
            return;
        }

        const timeoutId = setTimeout(() => {
            const container = ref?.current?.firstElementChild;

            if (!container || !isValidIndex(scrollToIndex)) {
                return;
            }

            const containerWidth = container.clientWidth;
            const childrenWidth = container.firstElementChild?.clientWidth ?? 1;
            const childrenHeight = container.firstElementChild?.clientHeight ?? 1;
            const childrenPerRow = Math.floor(containerWidth / childrenWidth);
            const targetRow = Math.floor(scrollToIndex / childrenPerRow);
            const scrollTo = (childrenHeight + gap) * targetRow;

            callback(scrollTo);
            // we don't want to scroll immediately
        }, delay);

        return () => {
            timeoutId && clearTimeout(timeoutId);
        };
    }, [scrollToIndex]);
};
