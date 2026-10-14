// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ActionButton, Text } from '@geti-ui/ui';
import { clsx } from 'clsx';
import { isNull } from 'lodash-es';

import { useStatusBar } from './status-bar-context';
import { ConnectionStatus, MainStatusState } from './status-bar.interface';

import classes from './status-bar.module.scss';

const CONNECTION_LABELS: Record<ConnectionStatus, string> = {
    connected: 'Connected',
    connecting: 'Connecting...',
    disconnected: 'Disconnected',
    failed: 'Connection failed',
};

const ProgressBar = ({ activeStatus }: { activeStatus: MainStatusState | null }) => {
    const isIndeterminate = activeStatus?.progress === undefined;

    if (isNull(activeStatus)) return null;

    return (
        <div
            className={clsx(
                classes.progressFill,
                classes[activeStatus.variant],
                isIndeterminate && classes.indeterminate
            )}
            style={{ '--progress': `${activeStatus.progress ?? 100}%` }}
        />
    );
};

const WebRTCStatus = ({ connection, hasActiveStatus }: { connection: ConnectionStatus; hasActiveStatus: boolean }) => {
    return (
        <div className={classes.connectionSlot}>
            <div className={clsx(classes.connectionDot, hasActiveStatus ? classes.neutral : classes[connection])} />
            <Text UNSAFE_className={classes.connectionText}>{CONNECTION_LABELS[connection]}</Text>
        </div>
    );
};

const MainStatus = ({ activeStatus }: { activeStatus: MainStatusState | null }) => (
    <div className={classes.mainStatusArea}>
        {!isNull(activeStatus) ? (
            <>
                <Text UNSAFE_className={classes.message}>{activeStatus.message}</Text>
                {activeStatus.detail && <Text UNSAFE_className={classes.detail}>{activeStatus.detail}</Text>}
                {activeStatus.isCancellable && activeStatus.onCancel && (
                    <ActionButton isQuiet onPress={activeStatus.onCancel} UNSAFE_className={classes.cancelButton}>
                        Cancel
                    </ActionButton>
                )}
            </>
        ) : (
            <Text UNSAFE_className={clsx(classes.message, classes.idleMessage)}>Idle</Text>
        )}
    </div>
);

export const StatusBar = () => {
    const { connection, activeStatus } = useStatusBar();
    const hasActiveStatus = activeStatus !== null;

    return (
        <div className={classes.statusBar}>
            <ProgressBar activeStatus={activeStatus} />

            <div className={clsx(classes.contentWrapper, hasActiveStatus && classes.hasBackground)}>
                <WebRTCStatus connection={connection} hasActiveStatus={hasActiveStatus} />

                <MainStatus activeStatus={activeStatus} />
            </div>
        </div>
    );
};
