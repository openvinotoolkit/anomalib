---
name: ui-test-utils
description: "Use when writing or updating Anomalib Studio UI component or hook tests that need the shared render/renderHook helpers, React Router paths or parameters, React Query, theme, URL-query, stream, suspense, or toast providers."
---

# Anomalib Studio UI Test Utilities

Use this skill when working on tests in `application/ui/` that render React components or hooks. The shared utilities are defined in `application/ui/tests/utils.ts`.

## Purpose

Import `render` and `renderHook` from the shared test utilities rather than directly from `@testing-library/react` when the unit under test depends on application context. The utilities build a memory router and mount these providers:

- `QueryClientProvider` with an isolated `QueryClient` by default.
- `ThemeProvider` for `@geti-ui/ui` components.
- `NuqsAdapter` for URL query-state hooks.
- `StreamConnectionProvider` for stream-dependent UI.
- `Suspense` with `IntelBrandedLoading` fallback for lazy components.
- `Toast` for toast assertions and feedback behavior.

The default route is `/projects/project-123/inspect`, matched by the default path `/projects/:projectId/inspect`.

## When To Use Each Helper

Use `render` for a component, dialog, or view that renders DOM output. Use `renderHook` for a React hook that reads provider state, route parameters, search parameters, or React Query state.

Keep direct Testing Library rendering only for genuinely provider-free units. Do not duplicate the provider stack in individual tests.

## Advantages

- Tests exercise the component in the same context expected by the Studio application.
- The memory router makes route and route-parameter behavior deterministic without browser navigation.
- A fresh query client prevents React Query cache state leaking between tests by default.
- Shared provider setup keeps tests short and ensures new tests follow the established application test environment.
- Toast, stream, theme, URL-query, and suspense dependencies are available without per-test boilerplate.

## Component Tests

Import the helper using the relative path from the test file to `application/ui/tests/utils.ts`:

```tsx
import { screen } from "@testing-library/react";

import { render } from "../../../../../tests/utils";
import { ProjectPanel } from "./project-panel.component";

it("shows the selected project", () => {
  render(<ProjectPanel />, { route: "/projects/project-456/inspect" });

  expect(screen.getByText("Project details")).toBeVisible();
});
```

Pass `route` when the component needs a specific location or `projectId`. The supplied route must match the selected `path`.

```tsx
render(<ProjectSettings />, {
  path: "/projects/:projectId/settings",
  route: "/projects/project-456/settings",
});
```

Use Testing Library's accessible queries and async helpers after rendering. For server-backed behavior, configure the existing MSW `server` handler before calling `render`.

```tsx
server.use(
  http.get("/api/projects/{project_id}", () => HttpResponse.json(project)),
);

render(<ProjectPanel />, { route: "/projects/project-456/inspect" });

expect(
  await screen.findByRole("heading", { name: project.name }),
).toBeVisible();
```

## Hook Tests

Use the wrapper-enabled helper rather than creating a custom provider wrapper in the test.

```tsx
import { waitFor } from "@testing-library/react";

import { renderHook } from "../../../../../tests/utils";
import { useActivePipelineStatus } from "./use-active-pipeline-status.hook";

it("reports an active project", async () => {
  server.use(
    http.get("/api/active-pipeline", () =>
      HttpResponse.json({ project_id: "project-456" }),
    ),
  );

  const { result } = renderHook(() => useActivePipelineStatus("project-123"));

  await waitFor(() => {
    expect(result.current.hasActiveProject).toBe(true);
  });
});
```

The helper accepts the same `route`, `path`, and `queryClient` options as `render`, together with normal Testing Library hook options such as `initialProps`.

```tsx
const queryClient = new QueryClient();
const { result, rerender } = renderHook(
  ({ projectId }) => useActivePipelineStatus(projectId),
  {
    initialProps: { projectId: "project-123" },
    queryClient,
    route: "/projects/project-123/inspect",
  },
);

rerender({ projectId: "project-456" });
```

Provide `queryClient` only when the test must seed, inspect, or deliberately share cache state. Otherwise, use the default isolated client.

## Test Hygiene

- Configure MSW responses with `server.use(...)` for API behavior; do not perform real network requests.
- Wait for asynchronous query or mutation results with `findBy...` or `waitFor`.
- Keep routes explicit when behavior depends on project identifiers or URL state.
- Clear any externally imported, shared query client in `beforeEach` when the code under test uses one instead of the client supplied by this helper.
- Do not add application providers around `render` or `renderHook`; the utility already owns that setup.
