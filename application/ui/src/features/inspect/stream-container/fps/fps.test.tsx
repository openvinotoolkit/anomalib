import { screen, waitFor } from '@testing-library/react';
import { cloneDeep } from 'lodash-es';
import { http } from 'src/api/utils';
import { server } from 'src/msw-node-setup';

import { getMockedMetrics } from '../../../../../mocks/mock-metrics';
import { getMockedPipeline } from '../../../../../mocks/mock-pipeline';
import { render } from '../../../../../tests/utils';
import { Fps } from './fps.component';

describe('Fps', () => {
    const renderFps = ({
        metricsConfig,
        pipelineConfig,
    }: {
        metricsConfig?: Partial<ReturnType<typeof getMockedMetrics>> | null;
        pipelineConfig?: Partial<ReturnType<typeof getMockedPipeline>> | null;
    } = {}) => {
        server.use(
            http.get('/api/projects/{project_id}/pipeline/metrics', ({ response }) =>
                response(200).json(getMockedMetrics(metricsConfig ? metricsConfig : {}))
            ),
            http.get('/api/projects/{project_id}/pipeline', ({ response }) =>
                response(200).json(getMockedPipeline(pipelineConfig ? pipelineConfig : {}))
            )
        );
        return render(<Fps projectId={'123'} />);
    };

    it('renders FPS value when metrics are available', async () => {
        const metricsConfig = cloneDeep(getMockedMetrics({}));
        metricsConfig.inference.latency.latest_ms = 25;

        renderFps({ metricsConfig });
        expect(await screen.findByText(/40/)).toBeVisible();
    });

    it('renders nothing if metrics are missing', async () => {
        renderFps({ pipelineConfig: { status: 'running' }, metricsConfig: {} });

        await waitFor(() => {
            expect(screen.queryByText(/FPS/i)).not.toBeInTheDocument();
        });
    });

    it('renders nothing if pipeline is not running', async () => {
        renderFps({ pipelineConfig: { status: 'active' }, metricsConfig: {} });

        await waitFor(() => {
            expect(screen.queryByText(/FPS/i)).not.toBeInTheDocument();
        });
    });
});
