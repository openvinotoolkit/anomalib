import { Checkbox, CheckboxGroup } from '@geti-ui/ui';

import { OutputFormat, SinkOutputFormats } from '../utils';

import classes from './output-formats.module.scss';

type OutputFormatsProps = {
    config?: SinkOutputFormats;
};

export const OutputFormats = ({ config = [] }: OutputFormatsProps) => {
    return (
        <CheckboxGroup
            isRequired
            label='Output Formats'
            name='output_formats'
            defaultValue={config}
            UNSAFE_className={classes.itemList}
        >
            <Checkbox name='output_formats' value={OutputFormat.PREDICTIONS}>
                Predictions
            </Checkbox>
            <Checkbox name='output_formats' value={OutputFormat.IMAGE_ORIGINAL}>
                Image Original
            </Checkbox>
            <Checkbox name='output_formats' value={OutputFormat.IMAGE_WITH_PREDICTIONS}>
                Image with Predictions
            </Checkbox>
        </CheckboxGroup>
    );
};
