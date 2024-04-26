// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

use crate::errors::ImageError;
use opencv::{
    core::Mat,
    imgcodecs::{imread, imwrite},
    imgproc::{apply_color_map, COLORMAP_JET},
    prelude::*,
    types::VectorOfi32,
};

pub struct Visualizer {
    image: Mat,
    output_path: String,
}

impl Visualizer {
    pub fn new(input_path: &str, output_path: &String) -> Result<Visualizer, ImageError> {
        let image = imread(input_path, opencv::imgcodecs::IMREAD_COLOR)?;

        Ok(Visualizer {
            image: image,
            output_path: output_path.to_string(),
        })
    }
    pub fn visualize(&self, model_result: &Vec<f32>) -> () {
        let heatmap = self
            .raw_to_colormap(model_result)
            .expect("Error superimposing heatmap");
        let superimposed = self.superimpose_heatmap(&heatmap);
        self.save(&superimposed);
    }

    fn raw_to_colormap(&self, model_result: &Vec<f32>) -> Result<Mat, ImageError> {
        let raw_heatmap = Mat::from_slice_rows_cols(&model_result, 256, 256)?;
        let mut scaled_heatmap = Mat::default();
        raw_heatmap.convert_to(&mut scaled_heatmap, opencv::core::CV_8UC1, 255.0, 0.0)?;
        let mut heatmap = Mat::default();
        apply_color_map(&scaled_heatmap, &mut heatmap, COLORMAP_JET)?;
        Ok(heatmap)
    }

    fn superimpose_heatmap(&self, heatmap: &Mat) -> Mat {
        let mut resized_heatmap = Mat::default();
        opencv::imgproc::resize(
            &heatmap,
            &mut resized_heatmap,
            opencv::core::Size::new(self.image.cols(), self.image.rows()),
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )
        .expect("Error resizing heatmap");
        let mut superimposed = Mat::default();
        opencv::core::add_weighted(
            &self.image,
            0.4,
            &resized_heatmap,
            0.6,
            0.0,
            &mut superimposed,
            opencv::core::CV_8UC3,
        )
        .expect("Error superimposing heatmap");
        superimposed
    }

    fn save(&self, image: &Mat) {
        imwrite(&self.output_path, &image, &VectorOfi32::new()).expect("Error saving image");
    }
}
