// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Flex } from '@geti-ui/ui';
import { Play } from '@geti-ui/ui/icons';
import { clsx } from 'clsx';
import { isEmpty } from 'lodash-es';
import { usePipeline } from 'src/hooks/use-pipeline.hook';

import classes from './play-stream-button.module.scss';

type PlayStreamButtonProps = {
    onStart?: () => void;
    isDisabled?: boolean;
};

export const PlayStreamButton = ({ isDisabled: isDisabledByProp = false, onStart }: PlayStreamButtonProps) => {
    const { data: pipeline } = usePipeline();

    const isDisabled = isDisabledByProp || isEmpty(pipeline?.source);

    return (
        <button
            type='button'
            onClick={onStart}
            disabled={isDisabled}
            aria-label={'Start stream'}
            className={clsx(classes.container, { [classes.disabled]: isDisabled })}
        >
            <Flex UNSAFE_className={classes.playButton}>
                <Play width='20px' height='20px' />
            </Flex>
        </button>
    );
};
