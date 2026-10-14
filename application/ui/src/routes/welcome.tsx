// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Content, Flex, Grid, Heading, IllustratedMessage, Text } from '@geti-ui/ui';
import { Navigate, useNavigate } from 'react-router';
import { v4 as uuid } from 'uuid';

import { $api } from '../api/client';
import Background from '../assets/background.png';
import { Fireworks } from '../assets/icons';
import { paths } from './paths';

const useCreateProject = () => {
    const createProjectMutation = $api.useMutation('post', '/api/projects', {
        meta: {
            invalidates: [['get', '/api/projects']],
        },
    });
    const navigate = useNavigate();

    const createProject = (projectName: string, projectId: string = uuid()) => {
        createProjectMutation.mutate(
            {
                body: {
                    id: projectId,
                    name: projectName,
                },
            },
            {
                onSuccess: () => {
                    navigate(`${paths.project({ projectId })}?mode=Dataset`);
                },
            }
        );
    };

    return createProject;
};

export const Welcome = () => {
    const createProject = useCreateProject();
    const { data } = $api.useSuspenseQuery('get', '/api/projects');

    if (data.projects.length > 0) {
        return <Navigate to={paths.root({})} replace />;
    }

    const handleCreateProject = () => {
        createProject('Project #1');
    };

    return (
        <Grid
            height={'100%'}
            minHeight='100vh'
            UNSAFE_style={{
                backgroundColor: 'var(--spectrum-global-color-gray-100)',
                backgroundImage: `url(${Background})`,
                backgroundBlendMode: 'luminosity',
                backgroundPosition: 'center',
                backgroundRepeat: 'no-repeat',
                backgroundSize: 'cover',
            }}
        >
            <IllustratedMessage>
                <Fireworks />
                <Heading level={1}>Welcome to Anomalib Studio!</Heading>
                <Content>
                    <Flex direction={'column'} gap={'size-200'}>
                        <Text>Press the button below to create your first project</Text>
                        <Button onPress={handleCreateProject}>Create project</Button>
                    </Flex>
                </Content>
            </IllustratedMessage>
        </Grid>
    );
};
