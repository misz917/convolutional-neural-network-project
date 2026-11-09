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
            tensor = layer.apply(tensor);
        }

        assert_eq!(tensor.channels(), 1);
        assert_eq!(tensor.height(), 1);

        let map = &tensor.get_maps()[0];

        map.clone().into_vec()
    }
}
