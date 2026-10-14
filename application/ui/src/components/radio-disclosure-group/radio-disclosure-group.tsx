import { ReactNode } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex, Radio, RadioGroup, View } from '@geti-ui/ui';

import classes from './radio-disclosure-group.module.scss';

export const RadioDisclosure = <ValueType extends string>({
    value,
    setValue,
    items,
    ariaLabel,
}: {
    value: ValueType;
    setValue: (value: ValueType) => void;
    items: Array<{
        value: ValueType;
        label: ReactNode;
        content: ReactNode;
    }>;
    ariaLabel?: string;
}) => {
    return (
        <RadioGroup
            orientation='vertical'
            width='100%'
            onChange={(newValue) => {
                setValue(newValue as ValueType);
            }}
            aria-label={ariaLabel}
            value={value}
        >
            <Flex direction='column' gap='size-200' minWidth={'size-6000'}>
                {items.map((item) => {
                    return (
                        <Disclosure
                            key={item.value}
                            onExpandedChange={(expanded) => expanded && setValue(item.value)}
                            isExpanded={item.value === value}
                            UNSAFE_className={classes.disclosure}
                        >
                            <DisclosureTitle UNSAFE_className={classes.disclosureTitle}>
                                <View padding='size-200'>
                                    <Radio value={item.value} UNSAFE_className={classes.radio}>
                                        <Flex alignItems='center' gap='size-200'>
                                            {item.label}
                                        </Flex>
                                    </Radio>
                                </View>
                            </DisclosureTitle>
                            <DisclosurePanel UNSAFE_className={classes.disclosurePanel}>
                                <View padding='size-200'>{item.content}</View>
                            </DisclosurePanel>
                        </Disclosure>
                    );
                })}
            </Flex>
        </RadioGroup>
    );
};
