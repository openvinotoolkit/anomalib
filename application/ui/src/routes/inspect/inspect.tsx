// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { Grid } from '@geti-ui/ui';

import { Footer } from '../../features/inspect/footer/footer.component';
import { MainContent } from '../../features/inspect/main-content/main-content.component';
import { Sidebar } from '../../features/inspect/sidebar.component';
import { Toolbar } from '../../features/inspect/toolbar/toolbar';

export const Inspect = () => {
    const { projectId } = useProjectIdentifier();

    return (
        <Grid
            areas={['toolbar sidebar', 'canvas sidebar', 'footer sidebar']}
            rows={['size-800', 'minmax(0, 1fr)', 'auto']}
            columns={['1fr', 'min-content']}
            height={'100%'}
            gap={'size-10'}
            UNSAFE_style={{
                overflow: 'hidden',
            }}
        >
            <Toolbar key={`toolbar-${projectId}`} />
            {/* do not refresh the stream to avoid duplicate connections */}
            <MainContent />
            <Sidebar key={`sidebar-${projectId}`} />
            <Footer key={`footer-${projectId}`} />
        </Grid>
    );
};
