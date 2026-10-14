import { Flex, Item, NumberField, Picker, TextField } from '@geti-ui/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitField } from '../rate-limit-field/rate-limit-field.component';
import { WebhookHttpMethod, WebhookSinkConfig } from '../utils';
import { HeaderKeyValueBuilder } from './header-key-value-builder.component';

interface WebhookFieldsProps {
    defaultState: WebhookSinkConfig;
}

export const WebhookFields = ({ defaultState }: WebhookFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField isRequired label='Name' name='name' defaultValue={defaultState.name} />

            <Flex direction={'row'} gap='size-200'>
                <Picker name='http_method' flex='1' label='HTTP Method' defaultSelectedKey={defaultState.http_method}>
                    <Item key={WebhookHttpMethod.POST}>{WebhookHttpMethod.POST}</Item>
                    <Item key={WebhookHttpMethod.PATCH}>{WebhookHttpMethod.PATCH}</Item>
                    <Item key={WebhookHttpMethod.PUT}>{WebhookHttpMethod.PUT}</Item>
                </Picker>
                <NumberField label='Timeout' name='timeout' minValue={1} step={1} defaultValue={defaultState.timeout} />
            </Flex>

            <TextField
                isRequired
                width={'100%'}
                label='Webhook URL'
                name='webhook_url'
                defaultValue={defaultState.webhook_url}
            />

            <RateLimitField defaultValue={defaultState.rate_limit} />

            <OutputFormats config={defaultState.output_formats} />

            <HeaderKeyValueBuilder
                config={defaultState.headers ?? undefined}
                title='Headers'
                keysName='headers-keys'
                valuesName='headers-values'
                key={JSON.stringify(defaultState.headers) + defaultState.id}
            />
        </Flex>
    );
};
