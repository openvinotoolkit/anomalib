import { useEffect, useRef } from 'react';

import { useUnwrapDOMRef } from '@geti-ui/ui';

interface UseListEndOptions {
    onEndReached: () => void;
    rootMargin?: number;
    disabled?: boolean;
}

export const useListEnd = ({ onEndReached, rootMargin = 100, disabled = false }: UseListEndOptions) => {
    const sentinelRef = useRef(null);
    const unWrapSentinelRef = useUnwrapDOMRef(sentinelRef);

    useEffect(() => {
        if (disabled) return;

        const sentinel = unWrapSentinelRef.current;
        if (!sentinel) return;

        const observer = new IntersectionObserver(
            (entries) => {
                const isVisible = entries[0].isIntersecting;
                if (isVisible) onEndReached();
            },
            { rootMargin: `${rootMargin}px` }
        );

        observer.observe(sentinel);

        return () => observer.disconnect();
    }, [onEndReached, rootMargin, disabled, unWrapSentinelRef]);

    return sentinelRef;
};
