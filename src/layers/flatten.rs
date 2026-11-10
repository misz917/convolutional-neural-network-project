use crate::{layers::Layer, tensor::Tensor};

pub struct Flatten;

impl Layer for Flatten {
    fn forward(&mut self, tensor: Tensor) -> Tensor {
        let maps = tensor.get_maps();
        let merged = maps.concat().into_boxed_slice();

        Tensor::from(merged)
    }

    fn backward(&mut self, gradient: Tensor) -> Tensor {
        todo!()
    }

    fn update(&mut self, learning_rate: f32) {
        todo!()
    }
}
