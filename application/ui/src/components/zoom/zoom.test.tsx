import { act, render, renderHook, screen } from '@testing-library/react';
import { vi } from 'vitest';

import { useContainerSize } from './use-container-size';
import { useSetZoom, useZoom, ZoomProvider } from './zoom';
import { ZoomTransform } from './zoom-transform';

vi.mock('./use-container-size', () => ({
    useContainerSize: vi.fn(),
}));

describe('Zoom', () => {
    it('Scales and translates content so that it fits the screen', () => {
        const screenSize = { width: 100, height: 500 };
        const contentSize = { width: 300, height: 200 };
        const expectedZoom = { translate: { x: -100, y: 150 }, scale: 0.3 };

        vi.mocked(useContainerSize).mockImplementation(() => screenSize);

        render(
            <ZoomProvider>
                <ZoomTransform target={contentSize}>Content</ZoomTransform>
            </ZoomProvider>
        );

        const transform = screen.getByTestId('zoom-transform');

        expect(transform).toHaveStyle({
            transform: `translate(${expectedZoom.translate.x}px, ${expectedZoom.translate.y}px) scale(${expectedZoom.scale})`,
        });
    });

    it('Clamps scale and translations to boundaries', () => {
        const { result } = renderHook(() => ({ zoom: useZoom(), setZoom: useSetZoom() }), { wrapper: ZoomProvider });

        // Zero / Negative Scale
        act(() => {
            result.current.setZoom({ scale: 0, translate: { x: 0, y: 0 } });
        });
        expect(result.current.zoom.scale).toBe(0.1);

        act(() => {
            result.current.setZoom({ scale: -5, translate: { x: 0, y: 0 } });
        });
        expect(result.current.zoom.scale).toBe(0.1);

        // High Scale
        act(() => {
            result.current.setZoom({ scale: 100, translate: { x: 0, y: 0 } });
        });
        expect(result.current.zoom.scale).toBe(10.0);

        // Far off-screen translations
        act(() => {
            result.current.setZoom({ scale: 1.0, translate: { x: 99999, y: -99999 } });
        });
        expect(result.current.zoom.translate).toEqual({ x: 5000, y: -5000 });

        // Functional updates
        // Since we explicitly overwrite the prev state, let's start from a base scale of 9.0
        act(() => {
            result.current.setZoom({ scale: 9.0, translate: { x: 0, y: 0 } });
        });
        act(() => {
            result.current.setZoom((prev: { scale: number; translate: { x: number; y: number } }) => ({
                ...prev,
                scale: prev.scale * 2,
            }));
        });
        expect(result.current.zoom.scale).toBe(10.0); // Clamped from 9.0 * 2 = 18.0 -> 10.0
    });
});
