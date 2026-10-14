/// <reference types="@rsbuild/core/types" />

import 'react';

import { type NavigateOptions } from 'react-router';

declare module '@adobe/react-spectrum' {
    interface RouterConfig {
        routerOptions: NavigateOptions;
    }
}

declare module 'react' {
    interface CSSProperties {
        [key: `--${string}`]: string | number;
    }
}
