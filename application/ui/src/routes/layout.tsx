import { Flex, Grid, Tabs, View } from '@geti-ui/ui';
import { Outlet, useLocation } from 'react-router';

import { ProjectsListPanel } from './../features/inspect/projects-management/projects-list-panel.component';

const Header = () => {
    return (
        <View backgroundColor={'gray-300'} gridArea={'header'}>
            <Flex height='100%' alignItems={'center'} justifyContent={'space-between'} marginX='1rem' gap='size-200'>
                <View marginEnd='size-200' maxWidth={'5ch'}>
                    <span>Anomalib Studio</span>
                </View>

                <ProjectsListPanel />
            </Flex>
        </View>
    );
};

const getFirstPathSegment = (path: string): string => {
    const segments = path.split('/');
    return segments.length > 1 ? `/${segments[1]}` : '/';
};

export const Layout = () => {
    const { pathname } = useLocation();

    return (
        <Tabs aria-label='Header navigation' selectedKey={getFirstPathSegment(pathname)}>
            <Grid
                areas={['header', 'content']}
                UNSAFE_style={{
                    gridTemplateRows: 'var(--spectrum-global-dimension-size-800, 4rem) minmax(0, 1fr)',
                }}
                minHeight={'100vh'}
                maxHeight={'100vh'}
                height={'100%'}
            >
                <Header />
                <View backgroundColor={'gray-50'} gridArea={'content'}>
                    <Outlet />
                </View>
            </Grid>
        </Tabs>
    );
};
