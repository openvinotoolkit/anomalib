// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

import { Button, Heading, IllustratedMessage, View } from '@geti-ui/ui';
import { NotFound } from '@geti-ui/ui/icons';
import { isRouteErrorResponse, useRouteError } from 'react-router';

import { paths } from '../../routes/paths';
import { redirectTo } from '../../routes/utils';

const useErrorMessage = () => {
    const error = useRouteError();

    if (isRouteErrorResponse(error)) {
        if (error.status === 400) {
            return 'The server cannot or will not process the current request.';
        }

        if (error.status === 403) {
            return 'You do not have permission to access this page.';
        }

        if (error.status === 404) {
            return "This page doesn't exist!";
        }

        if (error.status === 401) {
            return "You aren't authorized to see this";
        }

        if (error.status === 500) {
            return 'The server encountered an error and could not complete your request.';
        }

        if (error.status === 503) {
            return 'Looks like our API is down';
        }
    }

    if (error instanceof TypeError) {
        return error.message;
    }

    return 'An unknown error occurred';
};

export const ErrorPage = () => {
    const message = useErrorMessage();

    return (
        <View height={'100vh'}>
            <IllustratedMessage>
                <NotFound />
                <Heading>{message}</Heading>

                <Button
                    variant={'accent'}
                    marginTop={'size-200'}
                    onPress={() => {
                        redirectTo(paths.root({}));
                    }}
                >
                    Go back to home page
                </Button>
            </IllustratedMessage>
        </View>
    );
};
