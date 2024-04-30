// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

use opencv;
use opencv::core;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ImageError {
    #[error("Error loading image")]
    LoadError,
    #[error("Assertion Failed")]
    AssertionError,
    #[error("Undefined error code: {0}")]
    Undefined(i32),
}

impl From<opencv::Error> for ImageError {
    fn from(error: opencv::Error) -> Self {
        let error_code = error.code;
        match error_code {
            core::StsAssert => ImageError::AssertionError,
            _ => ImageError::Undefined(error_code),
        }
    }
}
