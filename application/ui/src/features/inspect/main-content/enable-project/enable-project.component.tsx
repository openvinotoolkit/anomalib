import { Suspense, useState } from 'react';

import { LinkExpired } from '@anomalib-studio/icons';
import { Button, DialogContainer, Flex, Loading, Text } from '@geti-ui/ui';

import { ConfirmationDialog } from './confirmation-dialog.component';

import classes from './enable-project.module.scss';

interface EnableProjectProps {
    activeProjectId: string;
    currentProjectId: string;
}

export const EnableProject = ({ activeProjectId, currentProjectId }: EnableProjectProps) => {
    const [isOpen, setIsOpen] = useState(false);

    return (
        <Flex UNSAFE_className={classes.container} alignItems={'center'} justifyContent={'center'}>
            <Flex direction='column' width={'90%'} maxWidth={'32rem'} gap={'size-200'} alignItems={'center'}>
                <LinkExpired />

                <Text UNSAFE_className={classes.description}>
                    This project is set as inactive, therefore the pipeline configuration is disabled for this project.
                    You can still explore the dataset and models within this inactive project.
                </Text>

                <Text UNSAFE_className={classes.title}>Would you like to activate this project?</Text>

                <Button onPress={() => setIsOpen(true)}>Activate project</Button>

                <DialogContainer onDismiss={() => setIsOpen(false)}>
                    {isOpen && (
                        <Suspense fallback={<Loading mode={'inline'} />}>
                            <ConfirmationDialog activeProjectId={activeProjectId} currentProjectId={currentProjectId} />
                        </Suspense>
                    )}
                </DialogContainer>
            </Flex>
        </Flex>
    );
};
