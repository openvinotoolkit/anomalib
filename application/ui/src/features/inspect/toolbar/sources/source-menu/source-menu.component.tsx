import { $api } from '@anomalib-studio/api';
import { useProjectIdentifier } from '@anomalib-studio/hooks';
import { toast } from '@anomalib-studio/toast';
import { ActionButton, Item, Key, Menu, MenuTrigger } from '@geti-ui/ui';
import { MoreMenu } from '@geti-ui/ui/icons';

export interface SourceMenuProps {
    id: string;
    name: string;
    isConnected: boolean;
    onEdit: () => void;
}

export const SourceMenu = ({ id, name, isConnected, onEdit }: SourceMenuProps) => {
    const { projectId } = useProjectIdentifier();

    const handleOnAction = (option: Key) => {
        switch (option) {
            case 'connect':
                handleConnect();
                break;
            case 'remove':
                handleDelete();
                break;
            default:
                onEdit();
                break;
        }
    };

    const updatePipeline = $api.useMutation('patch', '/api/projects/{project_id}/pipeline', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/sources', { params: { path: { project_id: projectId } } }],
                ['get', '/api/projects/{project_id}/pipeline', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    const handleConnect = async () => {
        try {
            await updatePipeline.mutateAsync({
                params: { path: { project_id: projectId } },
                body: { source_id: id },
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

    const removeSource = $api.useMutation('delete', '/api/projects/{project_id}/sources/{source_id}', {
        meta: {
            invalidates: [
                ['get', '/api/projects/{project_id}/sources', { params: { path: { project_id: projectId } } }],
            ],
        },
    });

    const handleDelete = async () => {
        try {
            await removeSource.mutateAsync({ params: { path: { project_id: projectId, source_id: id } } });

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
            <ActionButton isQuiet aria-label='source menu'>
                <MoreMenu />
            </ActionButton>
            <Menu onAction={handleOnAction} disabledKeys={isConnected ? ['connect', 'remove'] : []}>
                <Item key='connect'>Connect</Item>
                <Item key='edit'>Edit</Item>
                <Item key='remove'>Remove</Item>
            </Menu>
        </MenuTrigger>
    );
};
