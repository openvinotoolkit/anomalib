/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef, useState } from 'react';

import { Loading, TextField, type TextFieldRef } from '@geti-ui/ui';

interface ProjectEditionProps {
    name: string;
    isPending: boolean;
    onChange: (newName: string) => void;
}

export const ProjectEdition = ({ name, isPending, onChange }: ProjectEditionProps) => {
    const textFieldRef = useRef<TextFieldRef>(null);
    const [newName, setNewName] = useState<string>(name);

    const handleBlur = () => {
        onChange(newName);
    };

    const handleKeyDown = (event: React.KeyboardEvent) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            onChange(newName);
        }

        if (event.key === 'Escape') {
            event.preventDefault();
            setNewName(name);
            onChange(name);
        }
    };

    useEffect(() => {
        textFieldRef.current?.select();
    }, []);

    return (
        <>
            <TextField
                isQuiet
                ref={textFieldRef}
                value={newName}
                onBlur={handleBlur}
                onChange={setNewName}
                onKeyDown={handleKeyDown}
                isDisabled={isPending}
                aria-label='Edit project name'
            />
            {isPending && <Loading mode={'inline'} size='S' />}
        </>
    );
};
