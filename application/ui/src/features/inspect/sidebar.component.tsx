/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { Dataset as DatasetIcon, Models as ModelsIcon } from '@anomalib-studio/icons';
import { Flex, Grid, ToggleButton, View } from '@geti-ui/ui';
import { useSearchParams } from 'react-router';

import { Dataset } from './dataset/dataset.component';
import { FeedbackButton } from './feedback';
import { Models } from './models/models.component';

import styles from './sidebar.module.scss';

const TABS = [
    { label: 'Dataset', icon: <DatasetIcon />, content: <Dataset /> },
    { label: 'Models', icon: <ModelsIcon />, content: <Models /> },
    /* TODO: Add Stats tab implementation. See GitHub issue 
    https://github.com/open-edge-platform/anomalib/issues/3173 */
    /* { label: 'Stats', icon: <Stats />, content: <>Stats</> }, */
];

interface TabProps {
    tabs: (typeof TABS)[number][];
}

const SidebarTabs = ({ tabs }: TabProps) => {
    const [searchParams, setSearchParams] = useSearchParams();
    const selectTab = (tab: string | null) => {
        if (tab === null) {
            searchParams.delete('mode');
        } else {
            searchParams.set('mode', tab);
        }
        setSearchParams(searchParams);
    };
    const tab = searchParams.get('mode');

    const gridTemplateColumns = tab !== null ? ['clamp(size-4600, 35vw, 40rem)', 'size-600'] : ['0px', 'size-600'];

    const content = tabs.find(({ label }) => label === tab)?.content;

    return (
        <Grid
            gridArea={'sidebar'}
            UNSAFE_className={styles.container}
            columns={gridTemplateColumns}
            data-expanded={tab !== null}
            minHeight={0}
        >
            <View
                gridColumn={'1/2'}
                UNSAFE_className={styles.sidebarContent}
                backgroundColor={'gray-100'}
                paddingY={'size-200'}
                paddingX={'size-300'}
            >
                {content}
            </View>
            <View gridColumn={'2/3'} backgroundColor={'gray-200'} padding={'size-100'}>
                <Flex direction={'column'} height={'100%'} alignItems={'center'} gap={'size-100'}>
                    {tabs.map(({ label, icon }) => (
                        <ToggleButton
                            key={label}
                            isQuiet
                            isSelected={label === tab}
                            onChange={() => selectTab(label === tab ? null : label)}
                            UNSAFE_className={styles.toggleButton}
                            aria-label={`Toggle ${label} tab`}
                        >
                            {icon}
                        </ToggleButton>
                    ))}
                    <View marginTop={'auto'}>
                        <FeedbackButton />
                    </View>
                </Flex>
            </View>
        </Grid>
    );
};

export const Sidebar = () => {
    return <SidebarTabs tabs={TABS} />;
};
