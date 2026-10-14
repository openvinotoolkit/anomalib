import { NotFoundIcon } from '@anomalib-studio/icons';
import { Flex, Text } from '@geti-ui/ui';

import styles from './source-sink-message.module.scss';

export const SOURCE_MESSAGE = 'Setup your input source';

export const SourceSinkMessage = () => {
    return (
        <Flex
            gap={'size-125'}
            gridArea={'canvas'}
            direction={'column'}
            alignItems={'center'}
            alignContent={'center'}
            justifyContent={'center'}
            UNSAFE_className={styles.container}
        >
            <NotFoundIcon />
            <Text UNSAFE_className={styles.description}>
                Please select the data source that you want your model to infer on via the pipeline configuration panel
                on the right. Once connected, you can start to collect normal scenes from your data source to fine-tune
                your model.
            </Text>

            <Text UNSAFE_className={styles.title}>{SOURCE_MESSAGE}</Text>
        </Flex>
    );
};
