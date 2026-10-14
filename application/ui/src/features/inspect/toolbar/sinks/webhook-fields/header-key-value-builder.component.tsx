import { Fragment, useState } from 'react';

import { ActionButton, Content, ContextualHelp, dimensionValue, Flex, Grid, Text } from '@geti-ui/ui';
import { Add, Delete } from '@geti-ui/ui/icons';
import { isEmpty } from 'lodash-es';

import { RequiredTextField } from '../../../../../components/required-text-field/required-text-field.component';
import { Fields, getPairsFromObject, Pair } from './utils';

type HeaderKeyValueBuilderProps = {
    title: string;
    keysName: string;
    valuesName: string;
    config?: Record<string, string>;
};

const updatePairAtIndex = (indexToUpdate: number, field: Fields, value: string) => (pair: Pair, index: number) =>
    index === indexToUpdate ? { ...pair, [field]: value } : pair;

export const HeaderKeyValueBuilder = ({ title, keysName, valuesName, config = {} }: HeaderKeyValueBuilderProps) => {
    const [pairs, setPairs] = useState<Pair[]>(getPairsFromObject(config));

    const addPair = () => {
        setPairs([...pairs, { key: '', value: '' }]);
    };

    const updatePair = (indexToUpdate: number, field: Fields, value: string) => {
        setPairs((prevValues) => prevValues.map(updatePairAtIndex(indexToUpdate, field, value)));
    };

    const removePair = (indexToRemove: number) => {
        setPairs((prevValues) => prevValues.filter((_, index) => index !== indexToRemove));
    };

    return (
        <Flex direction='column' gap='size-100'>
            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Flex gap={'size-100'}>
                    <Text
                        UNSAFE_style={{
                            fontSize: dimensionValue('size-150'),
                            color: 'var(--spectrum-alias-label-text-color)',
                        }}
                    >
                        {title}
                    </Text>

                    <ContextualHelp variant='info'>
                        <Content>
                            <Text>
                                Add as many key-value pairs as needed. Each pair will be included in the &apos;{title}
                                &apos; object.
                            </Text>
                        </Content>
                    </ContextualHelp>
                </Flex>
                <ActionButton isQuiet onPress={addPair}>
                    <Add />
                </ActionButton>
            </Flex>

            <Grid columns={['1fr', '1fr', '50px']} gap={'size-100'}>
                {pairs.map((pair, index) => (
                    <Fragment key={`${keysName}-${index}`}>
                        <RequiredTextField
                            isQuiet
                            width={'100%'}
                            name={keysName}
                            value={pair.key}
                            aria-label={keysName}
                            placeholder='key'
                            errorMessage='Key cannot be empty'
                            onChange={(val) => updatePair(index, Fields.KEY, val)}
                        />
                        <RequiredTextField
                            isQuiet
                            width={'100%'}
                            name={valuesName}
                            value={pair.value}
                            aria-label={valuesName}
                            placeholder='value'
                            errorMessage='Value cannot be empty'
                            isDisabled={isEmpty(pair.key)}
                            onChange={(val) => updatePair(index, Fields.VALUE, val)}
                        />
                        <ActionButton aria-label='Remove' onPress={() => removePair(index)}>
                            <Delete />
                        </ActionButton>
                    </Fragment>
                ))}
            </Grid>
        </Flex>
    );
};
