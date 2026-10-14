// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useRef, useState } from 'react';

import { Flex, Text, View } from '@geti-ui/ui';

import { LogEntryComponent } from './log-entry.component';
import { LogFiltersComponent } from './log-filters.component';
import { DEFAULT_LOG_FILTERS, LogEntry, LogFilters } from './log-types';

import styles from './log-viewer.module.scss';

interface LogViewerProps {
    logs: LogEntry[];
    isLoading?: boolean;
}

const filterLogs = (logs: LogEntry[], filters: LogFilters): LogEntry[] => {
    return logs.filter((log) => {
        // Filter by level
        if (!filters.levels.has(log.record.level.name)) {
            return false;
        }

        // Filter by search query
        if (filters.searchQuery) {
            const query = filters.searchQuery.toLowerCase();
            const message = log.record.message.toLowerCase();
            const module = log.record.module.toLowerCase();
            const func = log.record.function.toLowerCase();

            if (!message.includes(query) && !module.includes(query) && !func.includes(query)) {
                return false;
            }
        }

        // Filter by time range
        if (filters.startTime !== null && log.record.time.timestamp < filters.startTime) {
            return false;
        }
        if (filters.endTime !== null && log.record.time.timestamp > filters.endTime) {
            return false;
        }

        return true;
    });
};

export const LogViewer = ({ logs, isLoading }: LogViewerProps) => {
    const [filters, setFilters] = useState<LogFilters>(DEFAULT_LOG_FILTERS);
    const [autoScroll, setAutoScroll] = useState(true);
    const logsContainerRef = useRef<HTMLDivElement | null>(null);

    const filteredLogs = useMemo(() => filterLogs(logs, filters), [logs, filters]);

    const handleFiltersChange = (newFilters: LogFilters) => {
        setFilters(newFilters);
    };

    useEffect(() => {
        if (!autoScroll || !logsContainerRef.current) {
            return;
        }
        logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }, [autoScroll, filteredLogs.length]);

    return (
        <View UNSAFE_className={styles.logViewer}>
            <LogFiltersComponent
                filters={filters}
                onFiltersChange={handleFiltersChange}
                totalCount={logs.length}
                filteredCount={filteredLogs.length}
                filteredLogs={filteredLogs}
                autoScroll={autoScroll}
                onAutoScrollChange={setAutoScroll}
            />

            <div className={styles.logsContainer} ref={logsContainerRef}>
                {filteredLogs.length === 0 ? (
                    <Flex UNSAFE_className={styles.emptyState}>
                        {isLoading ? (
                            <Text>Loading logs...</Text>
                        ) : logs.length === 0 ? (
                            <Text>No logs available</Text>
                        ) : (
                            <Text>No logs match the current filters</Text>
                        )}
                    </Flex>
                ) : (
                    <View UNSAFE_className={styles.logsInner}>
                        {filteredLogs.map((entry, idx) => (
                            <LogEntryComponent key={idx} entry={entry} />
                        ))}
                    </View>
                )}
            </div>
        </View>
    );
};
