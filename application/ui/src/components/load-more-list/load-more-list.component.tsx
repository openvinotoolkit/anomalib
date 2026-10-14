import { ReactNode } from 'react';

import { Button, dimensionValue, Flex } from '@geti-ui/ui';
import { useListEnd } from 'src/hooks/use-list-end.hook';

type LoadMoreListProps = {
    children: ReactNode;
    isLoading: boolean;
    hasNextPage: boolean;
    onLoadMore: () => void;
};

export const LoadMoreList = ({ children, isLoading, hasNextPage, onLoadMore }: LoadMoreListProps) => {
    const sentinelRef = useListEnd({ onEndReached: onLoadMore, disabled: isLoading || !hasNextPage });

    return (
        <Flex
            gap={'size-200'}
            maxHeight={'60vh'}
            direction={'column'}
            UNSAFE_style={{ overflow: 'auto', padding: dimensionValue('size-10') }}
        >
            {children}

            {hasNextPage && (
                <Button ref={sentinelRef} isDisabled={isLoading} isPending={isLoading} onPress={onLoadMore}>
                    Load More
                </Button>
            )}
        </Flex>
    );
};
