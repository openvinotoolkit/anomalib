// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

mod errors;
mod image;
mod model;
mod visualizer;
use image::Image;
use model::Model;
use visualizer::Visualizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!(
            "Usage: {} <image_path> <model_path.xml> [output_path]",
            args[0]
        );
        std::process::exit(1);
    }
    let image_path = &args[1];
    let model_path = &args[2];
    let mut output_path: &String = &String::from("output.png");
    if args.len() == 4 {
        output_path = &args[3];
    }
    let model = Model::new(model_path);

    let img = Image::new(image_path).unwrap();
    let visualizer = Visualizer::new(image_path, output_path).unwrap();

    let heatmap = model.infer(&img);
    visualizer.visualize(&heatmap);
}
