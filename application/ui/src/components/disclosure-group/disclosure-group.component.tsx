// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { ReactNode, useState } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex, Text } from '@geti-ui/ui';
import { clsx } from 'clsx';
import { isFunction } from 'lodash-es';

import styles from './disclosure-group.module.scss';

type DisclosureItem = { value: string; label: string; icon: ReactNode; content?: ReactNode };

type DisclosureItemProps = {
    value: string | null;
    onChange?: (value: string) => void;
    item: DisclosureItem;
};

interface DisclosureGroupProps {
    items: DisclosureItem[];
    defaultActiveInput: string | null;
}

const DisclosureItem = ({ item, value, onChange }: DisclosureItemProps) => {
    const isExpanded = item.value === value;

    const handleExpandedChange = () => {
        isFunction(onChange) && onChange(item.value);
    };

    return (
        <Disclosure
            isQuiet
            key={item.label}
            isExpanded={isExpanded}
            UNSAFE_className={clsx(styles.disclosure, { [styles.selected]: isExpanded })}
            onExpandedChange={handleExpandedChange}
        >
            <DisclosureTitle UNSAFE_className={styles.disclosureTitleContainer}>
                <Flex alignItems={'center'} justifyContent={'space-between'} width={'100%'}>
                    <Flex marginStart={'size-50'} alignItems={'center'} gap={'size-100'}>
                        {item.icon}
                        <Text UNSAFE_className={styles.disclosureTitle}>{item.label}</Text>
                    </Flex>
                </Flex>
            </DisclosureTitle>
            <DisclosurePanel>{isExpanded && item.content}</DisclosurePanel>
        </Disclosure>
    );
};

export const DisclosureGroup = ({ items, defaultActiveInput }: DisclosureGroupProps) => {
    const [activeInput, setActiveInput] = useState(defaultActiveInput);

    const handleActiveInputChange = (value: string) => {
        setActiveInput((prevValue) => (value !== prevValue ? value : null));
    };

    return (
        <Flex width={'100%'} direction={'column'} gap={'size-100'}>
            {items.map((item) => (
                <DisclosureItem item={item} key={item.label} onChange={handleActiveInputChange} value={activeInput} />
            ))}
        </Flex>
    );
};
