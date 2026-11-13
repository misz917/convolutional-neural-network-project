use std::time::Instant;

use image::DynamicImage;
use tracing::info;

use crate::{layers::Layer, tensor::Tensor};

pub struct CNN {
    layers: Vec<Box<dyn Layer>>,
}

impl CNN {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self { layers }
    }

    pub fn forward(&mut self, image: DynamicImage) -> Vec<f32> {
        let instant = Instant::now();
        info!("CNN started forward pass");
        let mut tensor = Tensor::from(&image);

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let instant = Instant::now();
            tensor = layer.forward(tensor);
            let elapsed = instant.elapsed();
            info!("Layer {i} processed in {}s", elapsed.as_secs_f32());
        }

        assert_eq!(tensor.channels(), 1);
        assert_eq!(tensor.height(), 1);

        let out = tensor.get_maps()[0].clone().into_vec();
        let elapsed = instant.elapsed();
        info!("CNN finished forward pass in {}s", elapsed.as_secs_f32());
        out
    }

    pub fn backward(&mut self, mut gradient: Tensor) {
        let instant = Instant::now();
        info!("CNN started backward pass");
        for (i, layer) in self.layers.iter_mut().rev().enumerate() {
            let instant = Instant::now();
            gradient = layer.backward(gradient);
            let elapsed = instant.elapsed();
            info!("Layer {i} processed in {}s", elapsed.as_secs_f32());
        }
        let elapsed = instant.elapsed();
        info!("CNN finished backward pass in {}s", elapsed.as_secs_f32());
    }

    pub fn update(&mut self, learning_rate: f32) {
        for layer in &mut self.layers {
            layer.update(learning_rate);
        }
    }
}
