import { getApiUrl } from '@anomalib-studio/api';
import { SchemaPredictionResponse } from '@anomalib-studio/api/spec';
import { DimensionValue, View } from '@geti-ui/ui';
import { clsx } from 'clsx';
import { motion } from 'motion/react';
import { ZoomProvider } from 'src/components/zoom/zoom';
import { ZoomTransform } from 'src/components/zoom/zoom-transform';
import { isNonEmptyString } from 'src/features/inspect/utils';

import { MediaItem } from '../../types';
import { useInference } from '../providers/inference-opacity-provider.component';
import { LabelScore } from './label-score.component';

import classes from './inference-result.module.scss';

interface InferenceResultProps {
    selectedMediaItem: MediaItem;
    inferenceResult: SchemaPredictionResponse | undefined;
}

const labelHeight: DimensionValue = 'size-350';

export const InferenceResult = ({ selectedMediaItem, inferenceResult }: InferenceResultProps) => {
    const { inferenceOpacity } = useInference();

    const size = {
        width: selectedMediaItem.width,
        height: selectedMediaItem.height,
    };

    return (
        <ZoomProvider>
            <ZoomTransform target={size}>
                <View height={'100%'} paddingTop={labelHeight}>
                    {inferenceResult && (
                        <View
                            top={0}
                            left={0}
                            height={labelHeight}
                            position={'absolute'}
                            maxWidth={'size-1600'}
                            UNSAFE_style={{
                                transform: `scale(calc(1 / var(--zoom-scale)))`,
                                transformOrigin: 'left bottom',
                            }}
                        >
                            <LabelScore label={inferenceResult.label} score={inferenceResult.score} />
                        </View>
                    )}

                    <View width={'100%'} height={'100%'} position={'relative'}>
                        <img
                            alt={selectedMediaItem.filename}
                            src={getApiUrl(
                                `/api/projects/${selectedMediaItem.project_id}/images/${selectedMediaItem.id}/full`
                            )}
                        />

                        {isNonEmptyString(inferenceResult?.anomaly_map) && (
                            <motion.img
                                exit={{ opacity: 0 }}
                                width={size.width}
                                height={size.height}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: inferenceOpacity }}
                                className={clsx(classes.inferenceImage)}
                                src={`data:image/png;base64,${inferenceResult.anomaly_map}`}
                                alt={`${selectedMediaItem.filename} inference`}
                            />
                        )}
                    </View>
                </View>
            </ZoomTransform>
        </ZoomProvider>
    );
};
