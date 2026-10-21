import { createContext, Dispatch, ReactNode, SetStateAction, useCallback, useContext, useState } from 'react';

type ZoomState = { scale: number; translate: { x: number; y: number } };

const MIN_SCALE = 0.1;
const MAX_SCALE = 10.0;
const MIN_TRANSLATE = -5000;
const MAX_TRANSLATE = 5000;

const clamp = (val: number, min: number, max: number): number => Math.min(Math.max(val, min), max);

const clampZoomState = (state: ZoomState): ZoomState => ({
    scale: clamp(state.scale, MIN_SCALE, MAX_SCALE),
    translate: {
        x: clamp(state.translate.x, MIN_TRANSLATE, MAX_TRANSLATE),
        y: clamp(state.translate.y, MIN_TRANSLATE, MAX_TRANSLATE),
    },
});
export const Zoom = createContext<ZoomState>({
    scale: 1.0,
    translate: { x: 0, y: 0 },
});
const SetZoom = createContext<Dispatch<SetStateAction<ZoomState>> | null>(null);

export const useZoom = () => {
    return useContext(Zoom);
};

export const useSetZoom = () => {
    const context = useContext(SetZoom);

    if (!context) {
        throw new Error('');
    }

    return context;
};

export const ZoomProvider = ({ children }: { children: ReactNode }) => {
    const [zoom, setZoom] = useState<ZoomState>({
        scale: 1.0,
        translate: { x: 0, y: 0 },
    });

    const setClampedZoom = useCallback<Dispatch<SetStateAction<ZoomState>>>((action) => {
        setZoom((prev) => {
            const next = isFunction(action) ? action(prev) : action;
            return clampZoomState(next);
        });
    }, []);

    return (
        <Zoom.Provider value={zoom}>
            <SetZoom.Provider value={setClampedZoom}>{children}</SetZoom.Provider>
        </Zoom.Provider>
    );
};
