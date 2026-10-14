// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedMediaItem } from 'mocks/mock-media-item';
import { TRAINABLE_MODELS } from 'mocks/mock-trainable-models';
import { HttpResponse } from 'msw';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { render } from '../../../../../tests/utils';
import { REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING } from '../../dataset/utils';
import { TrainModelButton } from './train-model-button.component';

const getMockedMediaItems = (count: number) => {
    return Array.from({ length: count }, (_, index) =>
        getMockedMediaItem({ id: `image-${index}`, project_id: 'project-123' })
    );
};

const mockImages = (count: number, total = count) => {
    server.use(
        http.get('/api/projects/{project_id}/images', () => {
            return HttpResponse.json({
                media: getMockedMediaItems(count),
                pagination: { offset: 0, limit: count, count, total },
            });
        })
    );
};

const mockTrainModelDialogResources = () => {
    server.use(
        http.get('/api/trainable-models', () => HttpResponse.json({ trainable_models: TRAINABLE_MODELS })),
        http.get('/api/system/devices/training', () =>
            HttpResponse.json([{ type: 'cpu', name: 'CPU', memory: null, index: null }])
        )
    );
};

describe('TrainModelButton', () => {
    it('disables the train button and explains how many normal images are missing', async () => {
        mockImages(REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING - 1);

        render(<TrainModelButton />);

        const button = await screen.findByRole('button', { name: /train model/i });

        expect(button).toBeDisabled();

        const infoButton = screen.getByRole('button', { name: /info/i });
        await userEvent.click(infoButton);

        expect(await screen.findByText('More images required')).toBeVisible();
        expect(screen.getByText('Add 1 more normal image to start training.')).toBeVisible();
    });

    it('enables the train button when enough normal images are uploaded', async () => {
        mockImages(REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING);
        mockTrainModelDialogResources();

        render(<TrainModelButton />);

        const button = await screen.findByRole('button', { name: /train model/i });

        await waitFor(() => {
            expect(button).not.toBeDisabled();
        });

        await userEvent.click(button);

        expect(await screen.findByRole('dialog')).toBeVisible();
        expect(await screen.findByText('PatchCore')).toBeVisible();
        expect(screen.getByRole('button', { name: /start/i })).toBeDisabled();
    });

    it('enables the train button when pagination total has enough normal images', async () => {
        mockImages(1, REQUIRED_NUMBER_OF_NORMAL_IMAGES_TO_TRIGGER_TRAINING);
        mockTrainModelDialogResources();

        render(<TrainModelButton />);

        const button = await screen.findByRole('button', { name: /train model/i });

        await waitFor(() => {
            expect(button).not.toBeDisabled();
        });
    });
});
