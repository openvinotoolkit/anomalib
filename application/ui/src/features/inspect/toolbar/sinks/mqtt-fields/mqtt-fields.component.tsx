import { Flex, NumberField, Switch, TextField } from '@geti-ui/ui';

import { OutputFormats } from '../output-formats/output-formats.component';
import { RateLimitField } from '../rate-limit-field/rate-limit-field.component';
import { MqttSinkConfig } from '../utils';

interface MqttFieldsProps {
    defaultState: MqttSinkConfig;
}

export const MqttFields = ({ defaultState }: MqttFieldsProps) => {
    return (
        <Flex direction='column' gap='size-200'>
            <TextField isHidden label='id' name='id' defaultValue={defaultState.id} />
            <TextField isHidden label='project_id' name='project_id' defaultValue={defaultState.project_id} />
            <TextField isRequired width='100%' label='Name' name='name' defaultValue={defaultState.name} />
            <TextField
                isRequired
                width='100%'
                label='Broker Host'
                name='broker_host'
                defaultValue={defaultState.broker_host}
            />

            <Flex direction='row' gap='size-200'>
                <TextField flex='1' label='Topic' name='topic' defaultValue={defaultState.topic} />
                <NumberField
                    label='Broker Port'
                    name='broker_port'
                    minValue={0}
                    step={1}
                    defaultValue={defaultState.broker_port}
                />
            </Flex>

            <Switch
                name='auth_required'
                aria-label='Require Authentication'
                defaultSelected={defaultState.auth_required}
                key={defaultState.auth_required ? 'true' : 'false'}
            >
                Auth Required
            </Switch>

            <RateLimitField defaultValue={defaultState.rate_limit} />

            <OutputFormats config={defaultState.output_formats} />
        </Flex>
    );
};
