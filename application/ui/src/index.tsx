import React from 'react';

import ReactDOM from 'react-dom/client';

import { Providers } from './providers';

import '@geti-ui/ui/styles.css';
import './index.css';

const rootEl = document.getElementById('root');
if (rootEl) {
    const root = ReactDOM.createRoot(rootEl);
    root.render(
        <React.StrictMode>
            <Providers />
        </React.StrictMode>
    );
}
