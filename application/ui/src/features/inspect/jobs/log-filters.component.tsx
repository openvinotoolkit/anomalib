// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import {
    ActionButton,
    Checkbox,
    Dialog,
    DialogTrigger,
    Flex,
    Icon,
    SearchField,
    Text,
    Tooltip,
    TooltipTrigger,
    View,
} from '@geti-ui/ui';
import { Copy, Filter } from '@geti-ui/ui/icons';

import { LOG_LEVEL_COLORS, LOG_LEVELS, LogEntry, LogFilters as LogFiltersType, LogLevelName } from './log-types';

import styles from './log-viewer.module.scss';

interface LevelCheckboxProps {
    level: LogLevelName;
    isSelected: boolean;
    onChange: (level: LogLevelName, selected: boolean) => void;
}

const LevelCheckboxItem = ({ level, isSelected, onChange }: LevelCheckboxProps) => {
    const color = LOG_LEVEL_COLORS[level];

    return (
        <label className={styles.levelMenuItem}>
            <input
                type='checkbox'
                checked={isSelected}
                onChange={(e) => onChange(level, e.target.checked)}
                className={styles.levelMenuCheckbox}
            />
            <span className={styles.levelMenuDot} style={{ backgroundColor: color }} />
            <span className={styles.levelMenuLabel}>{level}</span>
        </label>
    );
};

interface LevelDropdownProps {
    selectedLevels: Set<LogLevelName>;
    onLevelChange: (level: LogLevelName, selected: boolean) => void;
    onSelectAll: () => void;
    onClearAll: () => void;
}

const LevelDropdown = ({ selectedLevels, onLevelChange, onSelectAll, onClearAll }: LevelDropdownProps) => {
    const selectedCount = selectedLevels.size;
    const allSelected = selectedCount === LOG_LEVELS.length;
    const noneSelected = selectedCount === 0;

    return (
        <DialogTrigger type='popover'>
            <ActionButton aria-label='Filter by log level'>
                <Icon>
                    <Filter />
                </Icon>
                <Text>
                    Levels{' '}
                    <span className={styles.levelBadgeCount}>
                        {selectedCount}/{LOG_LEVELS.length}
                    </span>
                </Text>
            </ActionButton>
            <Dialog width='auto' UNSAFE_className={styles.levelDropdownDialog} UNSAFE_style={{ padding: 0 }}>
                <div className={styles.levelPopoverContent}>
                    {LOG_LEVELS.map((level) => (
                        <LevelCheckboxItem
                            key={level}
                            level={level}
                            isSelected={selectedLevels.has(level)}
                            onChange={onLevelChange}
                        />
                    ))}
                    <div className={styles.levelPopoverActions}>
                        <button onClick={onSelectAll} disabled={allSelected} className={styles.levelQuickButton}>
                            All
                        </button>
                        <button onClick={onClearAll} disabled={noneSelected} className={styles.levelQuickButton}>
                            None
                        </button>
                    </div>
                </div>
            </Dialog>
        </DialogTrigger>
    );
};

const formatLogForCopy = (log: LogEntry): string => {
    const timestamp = new Date(log.record.time.timestamp * 1000).toISOString();
    const level = log.record.level.name.padEnd(8);
    const source = `${log.record.module}:${log.record.function}:${log.record.line}`;
    return `[${timestamp}] ${level} ${source} - ${log.record.message}`;
};

interface LogFiltersProps {
    filters: LogFiltersType;
    onFiltersChange: (filters: LogFiltersType) => void;
    totalCount: number;
    filteredCount: number;
    filteredLogs: LogEntry[];
    autoScroll: boolean;
    onAutoScrollChange: (value: boolean) => void;
}

export const LogFiltersComponent = ({
    filters,
    onFiltersChange,
    totalCount,
    filteredCount,
    filteredLogs,
    autoScroll,
    onAutoScrollChange,
}: LogFiltersProps) => {
    const [copyStatus, setCopyStatus] = useState<'idle' | 'copied'>('idle');

    const handleCopyLogs = async () => {
        if (filteredLogs.length === 0) return;

        const formattedLogs = filteredLogs.map(formatLogForCopy).join('\n');

        try {
            await navigator.clipboard.writeText(formattedLogs);
            setCopyStatus('copied');
            setTimeout(() => setCopyStatus('idle'), 2000);
        } catch (err) {
            console.error('Failed to copy logs:', err);
        }
    };
    const handleLevelChange = (level: LogLevelName, selected: boolean) => {
        const newLevels = new Set(filters.levels);
        if (selected) {
            newLevels.add(level);
        } else {
            newLevels.delete(level);
        }
        onFiltersChange({ ...filters, levels: newLevels });
    };

    const handleSearchChange = (value: string) => {
        onFiltersChange({ ...filters, searchQuery: value });
    };

    const handleClearSearch = () => {
        onFiltersChange({ ...filters, searchQuery: '' });
    };

    const handleSelectAll = () => {
        onFiltersChange({ ...filters, levels: new Set(LOG_LEVELS) });
    };

    const handleClearAll = () => {
        onFiltersChange({ ...filters, levels: new Set() });
    };

    return (
        <View UNSAFE_className={styles.filtersContainer}>
            <Flex gap='size-150' alignItems='center' wrap='wrap'>
                <View UNSAFE_className={styles.searchContainer}>
                    <SearchField
                        aria-label='Search logs'
                        placeholder='Search logs...'
                        value={filters.searchQuery}
                        onChange={handleSearchChange}
                        onClear={handleClearSearch}
                        width='100%'
                    />
                </View>

                <LevelDropdown
                    selectedLevels={filters.levels}
                    onLevelChange={handleLevelChange}
                    onSelectAll={handleSelectAll}
                    onClearAll={handleClearAll}
                />

                <Checkbox isSelected={autoScroll} onChange={onAutoScrollChange} UNSAFE_className={styles.autoScroll}>
                    Auto-scroll
                </Checkbox>

                <TooltipTrigger delay={300}>
                    <ActionButton
                        aria-label='Copy logs to clipboard'
                        onPress={handleCopyLogs}
                        isDisabled={filteredCount === 0}
                    >
                        <Icon>
                            <Copy />
                        </Icon>
                        <Text>{copyStatus === 'copied' ? 'Copied!' : 'Copy'}</Text>
                    </ActionButton>
                    <Tooltip>Copy {filteredCount} logs to clipboard</Tooltip>
                </TooltipTrigger>

                <Text UNSAFE_className={styles.statsText}>
                    {filteredCount} / {totalCount}
                </Text>
            </Flex>
        </View>
    );
};
