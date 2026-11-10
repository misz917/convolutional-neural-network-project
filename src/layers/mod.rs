use crate::tensor::Tensor;

pub mod conv2d;
pub mod flatten;
pub mod linear;
pub mod maxpool2d;
pub mod relu;

pub trait Layer {
    fn forward(&mut self, tensor: Tensor) -> Tensor;
    fn backward(&mut self, gradient: Tensor) -> Tensor;
    fn update(&mut self, learning_rate: f32);
}
