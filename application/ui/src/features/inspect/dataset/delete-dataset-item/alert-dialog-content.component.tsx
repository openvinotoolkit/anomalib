import { AlertDialog, Text } from '@geti-ui/ui';

type AlertDialogContentProps = {
    itemsIds: string[];
    onPrimaryAction: () => void;
};

export const AlertDialogContent = ({ itemsIds, onPrimaryAction }: AlertDialogContentProps) => {
    return (
        <AlertDialog
            maxHeight={'size-6000'}
            title='Delete Items'
            variant='confirmation'
            primaryActionLabel='Confirm'
            secondaryActionLabel='Close'
            onPrimaryAction={onPrimaryAction}
            autoFocusButton='primary'
        >
            <Text>{`Are you sure you want to delete ${itemsIds.length} item(s)?`}</Text>
        </AlertDialog>
    );
};
