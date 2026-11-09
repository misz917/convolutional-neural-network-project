use crate::tensor::Tensor;

pub mod conv2d;
pub mod flatten;
pub mod linear;
pub mod maxpool2d;
pub mod relu;

pub trait Layer {
    fn apply(&self, tensor: Tensor) -> Tensor;
}
