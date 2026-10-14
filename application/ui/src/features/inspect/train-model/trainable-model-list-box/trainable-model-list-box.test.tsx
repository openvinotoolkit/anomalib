// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { RadioGroup, ThemeProvider } from '@geti-ui/ui';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TRAINABLE_MODELS } from 'mocks/mock-trainable-models';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { TrainableModelListBox } from './trainable-model-list-box.component';

describe('TrainableModelListBox', () => {
    const renderComponent = (selectedModelTemplateId: string | null = null) => {
        server.use(
            http.get('/api/trainable-models', ({ response }) =>
                response(200).json({ trainable_models: TRAINABLE_MODELS })
            )
        );

        return render(
            <QueryClientProvider client={new QueryClient()}>
                <ThemeProvider>
                    <RadioGroup aria-label='Select model' onChange={() => {}}>
                        <TrainableModelListBox selectedModelTemplateId={selectedModelTemplateId} />
                    </RadioGroup>
                </ThemeProvider>
            </QueryClientProvider>
        );
    };

    it('filters recommended models by default', async () => {
        renderComponent();

        expect(await screen.findByText('PatchCore')).toBeVisible();
        expect(await screen.findByText('Dinomaly')).toBeVisible();

        expect(screen.queryByText('PaDiM')).not.toBeInTheDocument();
        expect(screen.queryByText('STFPM')).not.toBeInTheDocument();
    });

    it('displays all models when "Show more" is clicked, hides them when "Show less" is clicked', async () => {
        renderComponent();

        expect(await screen.findByText('PatchCore')).toBeVisible();

        const showMoreButton = screen.getByRole('button', { name: /show more/i });
        await userEvent.click(showMoreButton);

        await waitFor(() => {
            expect(screen.getByText('PaDiM')).toBeVisible();
            expect(screen.getByText('STFPM')).toBeVisible();
        });

        const showLessButton = screen.getByRole('button', { name: /show less/i });
        await userEvent.click(showLessButton);

        await waitFor(() => {
            expect(screen.queryByText('PaDiM')).not.toBeInTheDocument();
            expect(screen.queryByText('STFPM')).not.toBeInTheDocument();
        });

        expect(screen.getByText('PatchCore')).toBeVisible();
    });
});
