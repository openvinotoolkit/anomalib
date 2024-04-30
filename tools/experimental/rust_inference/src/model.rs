// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

use crate::image::Image;
use std::fmt;

pub struct Model {
    model: openvino::Model,
    input_name: String,
    output_name: String,
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Model: input_name: {}, output_name: {}",
            self.input_name, self.output_name
        )
    }
}

impl Model {
    /// Create a new Model from the XML file at the given path.
    /// BIN path is taken from the XML file.
    ///
    /// # Examples
    ///
    /// ```rust
    /// let model = Model::new("model.xml");
    /// ```
    pub fn new(xml_path: &str) -> Model {
        // TODO should return Result
        let bin_path = xml_path.replace(".xml", ".bin");
        let mut core = openvino::Core::new().expect("Error creating Core");
        let network = core
            .read_model_from_file(&xml_path, &bin_path)
            .expect("Error reading model");
        let input_name = &network.get_input_by_index(0).unwrap();
        let output_name = &network.get_output_by_index(0).unwrap();
        Model {
            model: network,
            input_name: input_name.get_name().unwrap().to_string(),
            output_name: output_name.get_name().unwrap().to_string(),
        }
    }

    /// Infer image on the model.
    pub fn infer(&self, img: &Image) -> Vec<f32> {
        // TODO should return Result
        // let img_tensor = img.into();
        let mut compiled_model = self.model_from_image(&img);
        let mut infer_request = compiled_model
            .create_infer_request()
            .expect("Error creating InferRequest");

        let img: &Image = &Image {
            img: img.img.clone(),
        };

        let img_tensor = img.into();
        infer_request
            .set_tensor(&self.input_name, &img_tensor)
            .expect("Error setting tensor");
        infer_request.infer().expect("Error calling infer");
        let mut result = infer_request
            .get_tensor(&self.output_name)
            .expect("Error getting tensor");
        result
            .get_data::<f32>()
            .expect("Error getting data")
            .to_vec()
    }

    fn model_from_image(&self, img: &Image) -> openvino::CompiledModel {
        // TODO should return Result
        let ppp =
            openvino::PrePostProcess::new(&self.model).expect("Error creating PrePostProcess");

        let input_info = ppp
            .get_input_info_by_name(&self.input_name)
            .expect("Error getting input info");
        let mut input_tensor_info = input_info
            .preprocess_input_info_get_tensor_info()
            .expect("Error getting tensor info");
        input_tensor_info
            .preprocess_input_tensor_set_layout(
                &openvino::Layout::new("NHWC").expect("Error setting layout"),
            )
            .expect("Error setting layout");
        input_tensor_info
            .preprocess_input_tensor_set_from(&img.into())
            .expect("Error setting from");

        let model_info = input_info
            .get_model_info()
            .expect("Error getting model info");
        model_info
            .model_info_set_layout(&openvino::Layout::new("NCHW").expect("Error setting layout"))
            .expect("Error setting layout");

        let output_info = ppp
            .get_output_info_by_index(0)
            .expect("Error getting output info");
        output_info
            .get_output_info_get_tensor_info()
            .expect("Error getting tensor info")
            .preprocess_set_element_type(openvino::ElementType::F32)
            .expect("Error setting element type");

        let new_model = ppp.build_new_model().expect("Error building new model");
        let mut core = openvino::Core::new().expect("Error creating Core");
        core.compile_model(&new_model, "CPU")
            .expect("Error compiling model")
    }
}
