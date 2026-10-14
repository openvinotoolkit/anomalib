// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { getMockedPipeline } from 'mocks/mock-pipeline';
import { SchemaPipeline } from 'src/api/openapi-spec';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { render } from '../../../../../tests/utils';
import { PlayStreamButton } from './play-stream-button.component';

describe('PlayStreamButton', () => {
    const renderPlayStreamButton = ({
        pipelineConfig,
        onStart,
        isDisabled,
    }: {
        pipelineConfig?: Partial<SchemaPipeline>;
        onStart?: () => void;
        isDisabled?: boolean;
    } = {}) => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig))
            )
        );

        return render(<PlayStreamButton onStart={onStart} isDisabled={isDisabled} />);
    };

    it('calls onStart when clicked', async () => {
        const onStart = vi.fn();
        renderPlayStreamButton({ onStart });

        await userEvent.click(await screen.findByRole('button', { name: /Start stream/i }));

        expect(onStart).toHaveBeenCalledTimes(1);
    });

    it('renders in enabled state when pipeline has a source and isDisabled is false', async () => {
        renderPlayStreamButton();

        const button = await screen.findByRole('button', { name: /Start stream/i });
        expect(button.className).not.toMatch(/disabled/);
    });

    it('renders in disabled state when pipeline has no source', async () => {
        renderPlayStreamButton({ pipelineConfig: { source: null } });

        const button = await screen.findByRole('button', { name: /Start stream/i });
        expect(button.className).toMatch(/disabled/);
    });

    it('renders in disabled state when isDisabled prop is true', async () => {
        renderPlayStreamButton({ isDisabled: true });

        const button = await screen.findByRole('button', { name: /Start stream/i });
        expect(button.className).toMatch(/disabled/);
    });
});
