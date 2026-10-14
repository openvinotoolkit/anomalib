// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Content, ContextualHelp, Flex, Heading, Link, Radio, Text } from '@geti-ui/ui';
import { clsx } from 'clsx';

import { ModelTag } from './model-tag.component';
import { FAMILY_DISPLAY_NAMES, type TrainableModel } from './types';

import classes from './train-model.module.scss';

interface TrainableModelCardProps {
    model: TrainableModel;
    isSelected: boolean;
}

export const TrainableModelCard = ({ model, isSelected }: TrainableModelCardProps) => {
    return (
        <label
            htmlFor={`select-model-${model.id}`}
            aria-label={`Select ${model.name} model`}
            className={clsx(classes.modelCard, {
                [classes.modelCardSelected]: isSelected,
            })}
        >
            <Flex direction='column' gap='size-100' width='100%' height='100%'>
                <Flex alignItems='center' justifyContent='space-between'>
                    <Flex alignItems='center' gap='size-100'>
                        <Radio value={model.id} id={`select-model-${model.id}`} aria-label={model.name} />
                        <Text UNSAFE_className={classes.modelName}>{model.name}</Text>
                    </Flex>
                    {model.description && (
                        <ContextualHelp variant='info'>
                            <Heading>{model.name}</Heading>
                            <Content>
                                <Text>{model.description}</Text>
                            </Content>
                        </ContextualHelp>
                    )}
                </Flex>

                <Flex gap='size-75' wrap='wrap' flex={1}>
                    {model.recommended && <ModelTag label='Recommended' variant='recommended' />}
                    {model.family?.map((family) => (
                        <ModelTag key={family} label={FAMILY_DISPLAY_NAMES[family]} variant='info' />
                    ))}
                </Flex>

                <Flex direction='column' gap='size-100'>
                    <Flex direction='row' alignItems='baseline' gap='size-100'>
                        <Heading margin={0} UNSAFE_className={classes.attributeRatingTitle}>
                            Params
                        </Heading>
                        <Text UNSAFE_className={classes.attributeRatingValue}>{model.parameters}M</Text>
                    </Flex>

                    <Flex alignItems='center' justifyContent='space-between'>
                        <Text UNSAFE_className={classes.modelLicense}>{model.license}</Text>
                        {model.docs_url && (
                            <Link
                                href={model.docs_url}
                                target='_blank'
                                rel='noopener noreferrer'
                                UNSAFE_className={classes.docsLink}
                            >
                                <Text>Docs</Text>
                            </Link>
                        )}
                    </Flex>
                </Flex>
            </Flex>
        </label>
    );
};
