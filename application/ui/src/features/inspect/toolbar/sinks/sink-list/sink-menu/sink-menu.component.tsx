import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import { ActionButton, Item, Key, Menu, MenuTrigger } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

export interface SinkMenuProps {
    id: string;
    name: string;
    isConnected: boolean;
    onEdit: () => void;
}

export const SinkMenu = ({ id, name, isConnected, onEdit }: SinkMenuProps) => {
    const { projectId } = useProjectIdentifier();
    const removeSink = $api.useMutation('delete', '/api/projects/{project_id}/sinks/{sink_id}', {
        meta: {
            invalidates: [['get', '/api/projects/{project_id}/sinks', { params: { path: { project_id: projectId } } }]],
        },
    });

    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    const handleOnAction = (option: Key) => {
        switch (option) {
            case 'connect':
                handleConnect();
                break;
            case 'disconnect':
                handleDisconnect();
                break;
            case 'remove':
                handleDelete();
                break;
            default:
                onEdit();
                break;
        }
    };

    const handleConnect = async () => {
        try {
            await updatePipeline.mutateAsync({
                params: { path: { project_id: projectId } },
                body: { sink_id: id },
            });

            toast({
                type: 'success',
                message: `Successfully connected to "${name}"`,
            });
        } catch (_error) {
            toast({
                type: 'error',
                message: `Failed to connect to "${name}".`,
            });
        }
    };

    const handleDisconnect = async () => {
        try {
            await updatePipeline.mutateAsync({
                params: { path: { project_id: projectId } },
                body: { sink_id: null },
            });

            toast({
                type: 'success',
                message: `Successfully disconnected "${name}"`,
            });
        } catch (_error) {
            toast({
                type: 'error',
                message: `Failed to disconnect "${name}".`,
            });
        }
    };

    const handleDelete = async () => {
        try {
            await removeSink.mutateAsync({ params: { path: { sink_id: id, project_id: projectId } } });

            toast({
                type: 'success',
                message: `${name} has been removed successfully!`,
            });
        } catch (_error) {
            toast({
                type: 'error',
                message: `Failed to remove "${name}".`,
            });
        }
    };

    return (
        <MenuTrigger>
            <ActionButton isQuiet aria-label='sink menu'>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleOnAction} disabledKeys={isConnected ? ['connect', 'remove'] : ['disconnect']}>
                <Item key='connect'>Connect</Item>
                <Item key='disconnect'>Disconnect</Item>
                <Item key='edit'>Edit</Item>
                <Item key='remove'>Remove</Item>
            </Menu>
        </MenuTrigger>
    );
};
