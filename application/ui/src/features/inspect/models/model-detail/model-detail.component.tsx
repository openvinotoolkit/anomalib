// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { useState } from 'react';

import { Badge } from '@adobe/react-spectrum';
import { $api } from '@anomalib-studio/api';
import {
    ActionButton,
    Button,
    Content,
    ContextualHelp,
    Flex,
    Heading,
    Item,
    Link,
    Picker,
    Text,
    View,
    type Key,
} from '@geti-ui/ui';
import { Back } from '@geti-ui/ui/icons';
import type { SchemaCompressionType, SchemaExportType } from 'src/api/openapi-spec';
import { ActiveIcon, Onnx, OpenVino, PyTorch } from 'src/assets/icons';
import { useProjectIdentifier } from 'src/hooks/use-project-identifier.hook';

import type { ModelData } from '../../../../hooks/utils';
import { formatDuration, formatSize } from '../../utils';
import { useExportModel } from '../hooks/use-export-model.hook';

import classes from './model-detail.module.scss';

const openvinoDocsUrl =
    'https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/quantizing-models-post-training.html';

const EXPORT_FORMATS: { id: SchemaExportType; name: string; Icon: React.FC<React.SVGProps<SVGSVGElement>> }[] = [
    { id: 'openvino', name: 'OpenVINO', Icon: OpenVino },
    { id: 'onnx', name: 'ONNX', Icon: Onnx },
    { id: 'torch', name: 'PyTorch', Icon: PyTorch },
];

const COMPRESSION_OPTIONS: { id: SchemaCompressionType | 'none'; name: string }[] = [
    { id: 'none', name: 'None' },
    { id: 'fp16', name: 'FP16' },
    { id: 'int8', name: 'INT8' },
    { id: 'int8_ptq', name: 'INT8 PTQ' },
    { id: 'int8_acq', name: 'INT8 ACQ' },
];

interface ModelDetailProps {
    model: ModelData;
    isActiveModel: boolean;
    onBack: () => void;
}

export const ModelDetail = ({ model, isActiveModel, onBack }: ModelDetailProps) => {
    const { projectId } = useProjectIdentifier();
    const { data: project } = $api.useSuspenseQuery('get', '/api/projects/{project_id}', {
        params: { path: { project_id: projectId } },
    });

    const [selectedFormat, setSelectedFormat] = useState<SchemaExportType>('openvino');
    const [selectedCompression, setSelectedCompression] = useState<SchemaCompressionType | 'none'>('none');

    const exportModel = useExportModel();
    const isExportPending = exportModel.isPending || exportModel.isExporting(model.id);

    const handleFormatChange = (value: string) => {
        const format = value as SchemaExportType;
        setSelectedFormat(format);

        if (format !== 'openvino') {
            setSelectedCompression('none');
        }
    };

    const handleCompressionChange = (key: Key | null) => {
        if (key === null) return;
        setSelectedCompression(key as SchemaCompressionType | 'none');
    };

    const handleExport = () => {
        const compression = selectedCompression === 'none' ? null : selectedCompression;
        const formatLabel = EXPORT_FORMATS.find((f) => f.id === selectedFormat)?.name ?? selectedFormat;

        exportModel.mutate({
            projectId,
            projectName: project.name,
            modelId: model.id,
            modelName: model.name,
            format: selectedFormat,
            formatLabel,
            compression,
        });
    };

    return (
        <View UNSAFE_className={classes.container}>
            <Flex direction='column' gap='size-300'>
                <ActionButton isQuiet onPress={onBack} UNSAFE_className={classes.backButton}>
                    <Back />
                    <Text>Back to Models</Text>
                </ActionButton>

                <View UNSAFE_className={classes.modelHeader}>
                    <Flex direction='column' gap='size-200'>
                        <Flex alignItems='center' gap='size-200'>
                            <Text UNSAFE_className={classes.modelName}>{model.name}</Text>
                            {isActiveModel && (
                                <Badge variant='info' UNSAFE_className={classes.badge}>
                                    <ActiveIcon />
                                    Active
                                </Badge>
                            )}
                        </Flex>

                        <div className={classes.infoGrid}>
                            <div className={classes.infoItem}>
                                <span className={classes.infoLabel}>Training Date</span>
                                <span className={classes.infoValue}>{model.timestamp}</span>
                            </div>
                            <div className={classes.infoItem}>
                                <span className={classes.infoLabel}>Model Size</span>
                                <span className={classes.infoValue}>{formatSize(model.sizeBytes) || '-'}</span>
                            </div>
                            <div className={classes.infoItem}>
                                <span className={classes.infoLabel}>Training Duration</span>
                                <span className={classes.infoValue}>{formatDuration(model.durationInSeconds)}</span>
                            </div>
                            {model.backbone && (
                                <div className={classes.infoItem}>
                                    <span className={classes.infoLabel}>Model Backbone</span>
                                    <span className={classes.infoValue}>{model.backbone}</span>
                                </div>
                            )}
                        </div>
                    </Flex>
                </View>

                <View UNSAFE_className={classes.exportSection}>
                    <Flex direction='column' gap='size-200'>
                        <Text UNSAFE_className={classes.sectionTitle}>Export Model</Text>

                        <Flex direction='column' gap='size-100'>
                            <Text UNSAFE_className={classes.label}>Export Format</Text>
                            <div className={classes.formatGroup} role='radiogroup' aria-label='Select export format'>
                                {EXPORT_FORMATS.map(({ id, Icon }) => (
                                    <button
                                        key={id}
                                        type='button'
                                        role='radio'
                                        aria-checked={selectedFormat === id}
                                        onClick={() => handleFormatChange(id)}
                                        className={`${classes.formatOption} ${
                                            selectedFormat === id ? classes.formatOptionSelected : ''
                                        }`}
                                    >
                                        <Icon />
                                    </button>
                                ))}
                            </div>
                        </Flex>

                        <Flex direction='row' justifyContent='space-between' alignItems='end' height={'size-800'}>
                            {selectedFormat === 'openvino' && (
                                <Picker
                                    label='Compression (optional)'
                                    aria-label='Select compression type'
                                    items={COMPRESSION_OPTIONS}
                                    selectedKey={selectedCompression}
                                    onSelectionChange={handleCompressionChange}
                                    width='size-3000'
                                    contextualHelp={
                                        <ContextualHelp>
                                            <Heading>Compression Types</Heading>
                                            <Content>
                                                <Flex direction='column' gap='size-100'>
                                                    <Text>
                                                        <strong>FP16:</strong> Weight compression to FP16 precision. All
                                                        weights are converted to FP16.
                                                    </Text>
                                                    <Text>
                                                        <strong>INT8:</strong> Weight compression to INT8 precision. All
                                                        weights are quantized to INT8, but are dequantized to floating
                                                        point before inference.
                                                    </Text>
                                                    <Text>
                                                        <strong>INT8_PTQ:</strong> Full integer post-training
                                                        quantization to INT8 precision. All weights and operations are
                                                        quantized to INT8. Inference is performed in INT8 precision.
                                                    </Text>
                                                    <Text>
                                                        <strong>INT8_ACQ:</strong> Accuracy-control quantization to INT8
                                                        precision. Weights and operations are quantized to INT8, except
                                                        those that would degrade model quality beyond an acceptable
                                                        threshold. Inference uses mixed precision.
                                                    </Text>
                                                    <Text>
                                                        More info:{' '}
                                                        <Link
                                                            href={openvinoDocsUrl}
                                                            target='_blank'
                                                            rel='noopener noreferrer'
                                                        >
                                                            OpenVINO Quantization Documentation
                                                        </Link>
                                                    </Text>
                                                </Flex>
                                            </Content>
                                        </ContextualHelp>
                                    }
                                >
                                    {(item) => <Item key={item.id}>{item.name}</Item>}
                                </Picker>
                            )}

                            <Button
                                marginStart={'auto'}
                                variant='accent'
                                onPress={() => handleExport()}
                                isPending={isExportPending}
                            >
                                Export
                            </Button>
                        </Flex>
                    </Flex>
                </View>
            </Flex>
        </View>
    );
};
