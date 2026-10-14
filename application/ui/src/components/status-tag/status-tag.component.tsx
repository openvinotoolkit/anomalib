import { Flex, Text } from '@geti-ui/ui';
import { clsx } from 'clsx';

import classes from './status-tag.module.scss';

interface StatusTagProps {
    isError?: boolean;
    isConnected?: boolean;
}

export const StatusTag = ({ isConnected = false, isError = false }: StatusTagProps) => {
    if (isError) {
        return (
            <Flex gap={'size-75'} alignItems={'center'} UNSAFE_className={classes.container}>
                <div className={classes.status}></div>
                <Text>Error</Text>
            </Flex>
        );
    }
    return (
        <Flex gap={'size-75'} alignItems={'center'} UNSAFE_className={classes.container}>
            <div
                className={clsx({
                    [classes.status]: true,
                    [classes.connected]: isConnected,
                    [classes.disconnected]: !isConnected,
                })}
            ></div>
            <Text>{isConnected ? 'Connected' : 'Disconnected'}</Text>
        </Flex>
    );
};
