import { useState } from 'react';

import { Checkbox, Content, ContextualHelp, Flex, Heading, NumberField, Text } from '@geti-ui/ui';

interface RateLimitFieldProps {
    defaultValue: number | null | undefined;
}

const DEFAULT_RATE_LIMIT = 1;

export const RateLimitField = ({ defaultValue }: RateLimitFieldProps) => {
    const hasDefaultValue = defaultValue !== null && defaultValue !== undefined;
    const [isEnabled, setIsEnabled] = useState(hasDefaultValue || defaultValue === undefined);
    const [rateLimit, setRateLimit] = useState(defaultValue ?? DEFAULT_RATE_LIMIT);

    return (
        <Flex alignItems='center' gap='size-50'>
            <Checkbox isSelected={isEnabled} onChange={setIsEnabled}>
                Rate Limit:
            </Checkbox>
            <ContextualHelp variant='info'>
                <Heading>Rate limit</Heading>
                <Content>
                    <Text>
                        Maximum number of outputs per second (Hz). When disabled, outputs are sent without any rate
                        restriction.
                    </Text>
                </Content>
            </ContextualHelp>

            <input type='hidden' name='rate_limit' value={isEnabled ? rateLimit : ''} />

            {isEnabled && (
                <NumberField
                    aria-label='Rate limit value in Hz'
                    width='size-1200'
                    minValue={0.001}
                    step={0.1}
                    value={rateLimit}
                    onChange={setRateLimit}
                />
            )}
        </Flex>
    );
};
