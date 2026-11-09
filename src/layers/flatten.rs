use crate::{layers::Layer, tensor::Tensor};

pub struct Flatten;

impl Layer for Flatten {
    fn apply(&self, tensor: Tensor) -> Tensor {
        let maps = tensor.get_maps();
        let merged = maps.concat().into_boxed_slice();

        Tensor::from(merged)
    }
}
