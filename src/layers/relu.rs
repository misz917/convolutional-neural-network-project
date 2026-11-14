use crate::{layers::Layer, tensor::Tensor};

pub struct ReLU {
    cloned_output: Option<Tensor>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU {
            cloned_output: None,
        }
    }
}

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

        self.cloned_output = Some(tensor.clone());
        tensor
    }

    fn backward(&mut self, mut gradient: Tensor) -> Tensor {
        let cloned_output = self
            .cloned_output
            .as_ref()
            .expect("backward called before forward");
        let (width, height, channels) = gradient.dimensions();

        for c in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    if cloned_output.get_xyz(x, y, c) <= 0.0 {
                        gradient.set_xyz(x, y, c, 0.0);
                    }
                }
            }
        }

        gradient
    }

    fn update(&mut self, _learning_rate: f32) {
        // do nothing
    }
}
