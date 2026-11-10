use image::DynamicImage;

use crate::{layers::Layer, tensor::Tensor};

pub struct CNN {
    layers: Vec<Box<dyn Layer>>,
}

impl CNN {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, image: DynamicImage) -> Vec<f32> {
        let mut tensor = Tensor::from(&image);

        for layer in &mut self.layers {
            tensor = layer.forward(tensor);
        }

        assert_eq!(tensor.channels(), 1);
        assert_eq!(tensor.height(), 1);

        tensor.get_maps()[0].clone().into_vec()
    }

    pub fn backward(&mut self, mut gradient: Tensor) {
        for layer in self.layers.iter_mut().rev() {
            gradient = layer.backward(gradient);
        }
    }

    pub fn update(&mut self, learning_rate: f32) {
        for layer in &mut self.layers {
            layer.update(learning_rate);
        }
    }
}
