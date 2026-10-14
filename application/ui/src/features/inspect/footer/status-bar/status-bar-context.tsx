// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';

import { ConnectionStatus, MainStatusState, STATUS_PRIORITY } from './status-bar.interface';

interface StatusBarContextValue {
    /** WebRTC connection status */
    connection: ConnectionStatus;
    setConnection: (status: ConnectionStatus) => void;

    /** Map of all active statuses */
    statuses: Map<string, MainStatusState>;
    setStatus: (status: MainStatusState) => void;
    removeStatus: (id: string) => void;

    /** Get the highest priority active status */
    activeStatus: MainStatusState | null;
}

const StatusBarContext = createContext<StatusBarContextValue | null>(null);

interface StatusBarProviderProps {
    children: ReactNode;
}

export const StatusBarProvider = ({ children }: StatusBarProviderProps) => {
    const [connection, setConnection] = useState<ConnectionStatus>('disconnected');
    const [statuses, setStatuses] = useState<Map<string, MainStatusState>>(new Map());
    const autoRemoveTimers = useRef<Map<string, NodeJS.Timeout>>(new Map());

    const clearAutoRemoveTimer = useCallback((id: string) => {
        const timer = autoRemoveTimers.current.get(id);
        if (timer) {
            clearTimeout(timer);
            autoRemoveTimers.current.delete(id);
        }
    }, []);

    const removeStatus = useCallback(
        (id: string) => {
            clearAutoRemoveTimer(id);
            setStatuses((prev) => {
                const next = new Map(prev);
                next.delete(id);
                return next;
            });
        },
        [clearAutoRemoveTimer]
    );

    const setStatus = useCallback(
        (status: MainStatusState) => {
            clearAutoRemoveTimer(status.id);
            setStatuses((prev) => new Map(prev).set(status.id, status));

            if (status.autoRemoveDelay) {
                const timer = setTimeout(() => {
                    removeStatus(status.id);
                }, status.autoRemoveDelay);
                autoRemoveTimers.current.set(status.id, timer);
            }
        },
        [clearAutoRemoveTimer, removeStatus]
    );

    useEffect(() => {
        return () => {
            autoRemoveTimers.current.forEach((timer) => clearTimeout(timer));
        };
    }, []);

    // Get highest priority status (lowest priority number)
    const activeStatus = useMemo(() => {
        if (statuses.size === 0) return null;

        return Array.from(statuses.values()).sort((a, b) => STATUS_PRIORITY[a.type] - STATUS_PRIORITY[b.type])[0];
    }, [statuses]);

    const value = useMemo(
        () => ({
            connection,
            setConnection,
            statuses,
            setStatus,
            removeStatus,
            activeStatus,
        }),
        [connection, statuses, setStatus, removeStatus, activeStatus]
    );

    return <StatusBarContext.Provider value={value}>{children}</StatusBarContext.Provider>;
};

export const useStatusBar = () => {
    const context = useContext(StatusBarContext);
    if (!context) {
        throw new Error('useStatusBar must be used within StatusBarProvider');
    }
    return context;
};
