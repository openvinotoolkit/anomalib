// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Text, View } from '@geti-ui/ui';
import { clsx } from 'clsx';

import classes from './train-model.module.scss';

interface ModelTagProps {
    label: string;
    variant: 'recommended' | 'info';
}

export const ModelTag = ({ label, variant }: ModelTagProps) => {
    return (
        <View
            UNSAFE_className={clsx(classes.tag, {
                [classes.tagRecommended]: variant === 'recommended',
                [classes.tagInfo]: variant === 'info',
            })}
        >
            <Text UNSAFE_className={classes.tagText}>{label}</Text>
        </View>
    );
};
