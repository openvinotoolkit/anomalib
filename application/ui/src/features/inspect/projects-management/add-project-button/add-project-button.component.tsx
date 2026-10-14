/**
 * Copyright (C) 2025-2026 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { $api } from '@anomalib-studio/api';
import { ActionButton, Text } from '@geti-ui/ui';
import { AddCircle } from '@geti-ui/ui/icons';
import { v4 as uuid } from 'uuid';

import styles from './add-project-button.module.scss';

interface AddProjectProps {
    onSetProjectInEdition: (projectId: string) => void;
    projectsCount: number;
}

export const AddProjectButton = ({ onSetProjectInEdition, projectsCount }: AddProjectProps) => {
    const addProjectMutation = $api.useMutation('post', '/api/projects', {
        meta: {
            invalidates: [['get', '/api/projects']],
        },
    });

    const addProject = () => {
        const newProjectId = uuid();
        const newProjectName = `Project #${projectsCount + 1}`;

        addProjectMutation.mutate({
            body: {
                id: newProjectId,
                name: newProjectName,
            },
        });

        onSetProjectInEdition(newProjectId);
    };

    return (
        <ActionButton
            isQuiet
            width={'100%'}
            marginStart={'size-100'}
            marginEnd={'size-350'}
            UNSAFE_className={styles.addProjectButton}
            onPress={addProject}
        >
            <AddCircle />
            <Text marginX='size-50'>Add project</Text>
        </ActionButton>
    );
};
