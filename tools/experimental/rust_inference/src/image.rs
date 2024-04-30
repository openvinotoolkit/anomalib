// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

use crate::errors::ImageError;
use opencv::core::{Mat, MatTraitConst};
use opencv::imgcodecs::imread;
use openvino::{ElementType, Shape, Tensor};
use std::slice;
pub struct Image {
    pub(crate) img: Mat, // TODO remove pub
}

impl Image {
    /// Create a new Image from the image file at the given path.
    /// The image is converted to RGB format and is normalized to 0-1.
    pub fn new(path: &str) -> Result<Image, ImageError> {
        // TODO should return Result
        let img: Mat = imread(path, opencv::imgcodecs::IMREAD_COLOR)?;

        let mut img_rgb: Mat = Mat::default();
        opencv::imgproc::cvt_color(&img, &mut img_rgb, opencv::imgproc::COLOR_BGR2RGB, 0)?;

        let mut img_normalized: Mat = Mat::default();
        img_rgb.convert_to(
            &mut img_normalized,
            opencv::core::CV_32FC3,
            1.0 / 255.0,
            0.0,
        )?;
        Ok(Image {
            img: img_normalized,
        })
    }

    /// Returns a pointer to the image data.
    pub fn data(&self) -> *const u8 {
        self.img.data()
    }

    /// Returns the number of bytes in the image.
    /// Each pixel is represented by 4 bytes (fp32).
    pub fn bytes(&self) -> usize {
        let dimensions = self.img.size().unwrap();
        (dimensions.height * dimensions.width * self.img.channels() * 4) as usize
    }

    pub fn height(&self) -> i32 {
        self.img.size().unwrap().height
    }
    pub fn width(&self) -> i32 {
        self.img.size().unwrap().width
    }
    pub fn channels(&self) -> i32 {
        self.img.channels()
    }
}

impl From<&Image> for Tensor {
    fn from(img: &Image) -> Tensor {
        let img_slice = unsafe { slice::from_raw_parts(img.data(), img.bytes()) };
        let img_slice = img_slice.to_vec();
        let img_shape = Shape::new(&[
            1,
            img.height() as i64,
            img.width() as i64,
            img.channels() as i64,
        ])
        .expect("Error creating shape");
        let tensor = Tensor::new_from_host_ptr(ElementType::F32, &img_shape, &img_slice)
            .expect("Error creating tensor");
        tensor
    }
}
