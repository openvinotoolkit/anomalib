/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';

import {
    ActionButton,
    ButtonGroup,
    Content,
    Dialog,
    DialogTrigger,
    Divider,
    Flex,
    Header,
    Heading,
    View,
} from '@geti-ui/ui';

import { AddProjectButton } from './add-project-button/add-project-button.component';
import { useGetProjects } from './hooks/use-get-project.hooks';
import { useSelectedProject } from './hooks/use-selected-project.hook';
import { ProjectThumbnail } from './project-thumbnail/project-thumbnail.component';
import { ProjectsList } from './projects-list.component';

import styles from './projects-list.module.scss';

interface SelectedProjectProps {
    name: string;
    id: string | undefined;
}

const SelectedProjectButton = ({ name, id }: SelectedProjectProps) => {
    return (
        <ActionButton aria-label={`Selected project ${name}`} isQuiet height={'max-content'} staticColor='white'>
            <View margin={'size-50'}>{name}</View>
            <View margin='size-50'>
                <ProjectThumbnail projectId={id} projectName={name} size='size-400' />
            </View>
        </ActionButton>
    );
};

export const ProjectsListPanel = () => {
    const selectedProject = useSelectedProject();
    const [projectInEdition, setProjectInEdition] = useState<string | null>(null);
    const { projects, isFetchingNextPage, hasNextPage, fetchNextPage } = useGetProjects();

    const selectedProjectName = selectedProject?.name ?? '';

    return (
        <DialogTrigger type='popover' hideArrow>
            <SelectedProjectButton name={selectedProjectName} id={selectedProject?.id} />

            <Dialog width={'size-4600'} UNSAFE_className={styles.dialog}>
                <Header>
                    <Flex direction={'column'} justifyContent={'center'} width={'100%'} alignItems={'center'}>
                        <ProjectThumbnail
                            projectId={selectedProject?.id}
                            projectName={selectedProjectName}
                            size='size-1000'
                        />
                        <Heading level={2} marginBottom={0}>
                            {selectedProjectName}
                        </Heading>
                    </Flex>
                </Header>
                <Content>
                    <Divider size={'S'} marginY={'size-200'} />
                    <ProjectsList
                        projects={projects}
                        isLoading={isFetchingNextPage}
                        hasNextPage={hasNextPage}
                        onLoadMore={fetchNextPage}
                        projectIdInEdition={projectInEdition}
                        setProjectInEdition={setProjectInEdition}
                    />
                    <Divider size={'S'} marginY={'size-200'} />
                </Content>

                <ButtonGroup UNSAFE_className={styles.panelButtons}>
                    <AddProjectButton onSetProjectInEdition={setProjectInEdition} projectsCount={projects.length} />
                </ButtonGroup>
            </Dialog>
        </DialogTrigger>
    );
};
