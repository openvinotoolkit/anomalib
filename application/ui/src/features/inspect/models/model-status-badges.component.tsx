import { Badge } from '@adobe/react-spectrum';
import { ActiveIcon, LoadingIcon } from '@anomalib-studio/icons';
import { Alert, Cancel, Pending } from '@geti-ui/ui/icons';
import { SchemaJob } from 'src/api/openapi-spec';

import classes from './models-view.module.scss';

interface ModelStatusBadgesProps {
    isSelected: boolean;
    jobStatus?: SchemaJob['status'];
    progress?: number;
}

export const ModelStatusBadges = ({ isSelected, jobStatus, progress }: ModelStatusBadgesProps) => {
    if (!isSelected && !jobStatus) {
        return null;
    }

    return (
        <>
            {isSelected && (
                <Badge variant='info' UNSAFE_className={classes.badge}>
                    <ActiveIcon />
                    Active
                </Badge>
            )}
            {jobStatus === 'pending' && (
                <Badge variant='neutral' UNSAFE_className={classes.badge}>
                    <Pending />
                    Pending
                </Badge>
            )}
            {jobStatus === 'running' && (
                <Badge variant='info' UNSAFE_className={classes.badge}>
                    <LoadingIcon />
                    Training{progress !== undefined ? ` ${Math.round(progress)}%` : '...'}
                </Badge>
            )}
            {jobStatus === 'failed' && (
                <Badge variant='negative' UNSAFE_className={classes.badge}>
                    <Alert />
                    Failed
                </Badge>
            )}
            {jobStatus === 'canceled' && (
                <Badge variant='neutral' UNSAFE_className={classes.badge}>
                    <Cancel />
                    Canceled
                </Badge>
            )}
        </>
    );
};
