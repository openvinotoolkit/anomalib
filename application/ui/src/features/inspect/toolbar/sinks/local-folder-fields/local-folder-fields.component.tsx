import { useEffect, useId, useState } from 'react';

import { API_BASE_URL } from '@anomalib-studio/api';
import { Flex, TextField, View } from '@geti-ui/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitField } from '../rate-limit-field/rate-limit-field.component';
import { LocalFolderSinkConfig } from '../utils';

import styles from './local-folder-fields.module.scss';

interface LocalFolderFieldsProps {
    defaultState: LocalFolderSinkConfig;
}

const SINKS_SUFFIX = '/sinks/';

/**
 * Extract the suffix from a folder path by removing the {dataPath}/sinks/ prefix.
 * For new sinks, folder_path is just the suffix.
 * For existing sinks, folder_path contains the full path.
 */
const extractSuffix = (folderPath: string, dataPath: string): string => {
    if (!dataPath || !folderPath) return folderPath;
    const prefix = `${dataPath}${SINKS_SUFFIX}`;
    if (folderPath.startsWith(prefix)) {
        return folderPath.slice(prefix.length);
    }
    return folderPath;
};

export const LocalFolderFields = ({ defaultState }: LocalFolderFieldsProps) => {
    const folderPathLabelId = useId();
    const [dataPath, setDataPath] = useState<string>('');
    const [folderSuffix, setFolderSuffix] = useState(defaultState.folder_path);
    const [initializedSuffix, setInitializedSuffix] = useState(false);

    useEffect(() => {
        const fetchDataPath = async () => {
            try {
                const url = API_BASE_URL
                    ? new URL('/api/system/datapath', API_BASE_URL).toString()
                    : '/api/system/datapath';
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error(`Failed to fetch data path (HTTP ${response.status})`);
                }

                const raw = await response.text();
                // Endpoint may return a JSON string (quoted) or plain text.
                const parsed = (() => {
                    try {
                        return JSON.parse(raw) as unknown;
                    } catch {
                        return raw;
                    }
                })();

                setDataPath(typeof parsed === 'string' ? parsed : String(parsed));
            } catch (error) {
                console.error('Failed to fetch data path:', error);
            }
        };
        fetchDataPath();
    }, []);

    // Once we have the dataPath, extract the suffix from the folder_path (for editing existing sinks)
    useEffect(() => {
        if (dataPath && !initializedSuffix) {
            setFolderSuffix(extractSuffix(defaultState.folder_path, dataPath));
            setInitializedSuffix(true);
        }
    }, [dataPath, defaultState.folder_path, initializedSuffix]);

    const folderPrefix = dataPath ? `${dataPath}${SINKS_SUFFIX}` : `...${SINKS_SUFFIX}`;

    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            {/* Hidden field to store the full folder path for form submission */}
            <TextField isHidden label='folder_path' name='folder_path' value={`${folderPrefix}${folderSuffix}`} />

            <TextField isRequired label='Name' name='name' defaultValue={defaultState.name} />

            <View>
                <span id={folderPathLabelId} className={styles.folderPathLabel}>
                    Folder Path
                </span>
                <Flex direction='row' alignItems='center' gap='size-0'>
                    <View
                        backgroundColor='gray-200'
                        paddingX='size-150'
                        paddingY='size-100'
                        borderTopStartRadius='regular'
                        borderBottomStartRadius='regular'
                        UNSAFE_className={styles.folderPrefix}
                    >
                        <span className={styles.folderPrefixText}>{folderPrefix}</span>
                    </View>
                    <TextField
                        isRequired
                        flex='1'
                        aria-labelledby={folderPathLabelId}
                        value={folderSuffix}
                        onChange={setFolderSuffix}
                        UNSAFE_className={styles.folderSuffixField}
                    />
                </Flex>
            </View>

            <RateLimitField defaultValue={defaultState.rate_limit} />

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};
