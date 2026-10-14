import { Image } from '@anomalib-studio/icons';
import { Flex } from '@geti-ui/ui';

import styles from './dataset-item-placeholder.module.scss';

export const DatasetItemPlaceholder = () => {
    return (
        <Flex justifyContent={'center'} alignItems={'center'} UNSAFE_className={styles.datasetItemPlaceholder}>
            <Image />
        </Flex>
    );
};
