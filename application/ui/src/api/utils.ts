import { fromOpenApi } from '@msw/source/open-api';
import { createOpenApiHttp, OpenApiHttpHandlers } from 'openapi-msw';

import { paths } from './openapi-spec';
import spec from './openapi-spec.json' with { type: 'json' };

const handlers = await fromOpenApi(JSON.stringify(spec).replace(/}:/g, '}//:'));

const getOpenApiHttp = (baseUrl?: string): OpenApiHttpHandlers<paths> => {
    const http = createOpenApiHttp<paths>({
        baseUrl: baseUrl ?? process.env.PUBLIC_API_BASE_URL ?? 'http://localhost:8000',
    });

    // Escape every literal ":" in the OpenAPI path so path-to-regexp (used by MSW)
    // treats action suffixes like `/pipeline:activate` as literal characters
    // instead of URL parameters. Otherwise `/pipeline:activate` and `/pipeline:run`
    // collapse to the same pattern and the first-registered handler swallows both.
    // @see https://github.com/mswjs/msw/discussions/739
    const escapeActionColons = <P extends string>(path: P): P => path.replaceAll(':', '\\:') as P;

    return {
        ...http,
        post: (path, ...other) => http.post(escapeActionColons(path), ...other),
    };
};

const http = getOpenApiHttp();

export { getOpenApiHttp, handlers, http };
