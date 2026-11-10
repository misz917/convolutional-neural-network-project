use crate::{layers::Layer, tensor::Tensor};

pub struct ReLU;

impl Layer for ReLU {
    fn forward(&mut self, mut tensor: Tensor) -> Tensor {
        let width = tensor.width();
        let height = tensor.height();
        let channels = tensor.channels();

        for c in 0..channels {
            for x in 0..width {
                for y in 0..height {
                    if tensor.get_xyz(x, y, c) < 0.0 {
                        tensor.set_xyz(x, y, c, 0.0);
                    }
                }
            }
        }

        tensor
    }

    fn backward(&mut self, gradient: Tensor) -> Tensor {
        todo!()
    }

    fn update(&mut self, learning_rate: f32) {
        todo!()
    }
}
