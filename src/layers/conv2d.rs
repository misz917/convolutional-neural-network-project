use crate::{layers::Layer, tensor::Tensor};

pub struct Conv2D {
    kernels: Vec<Tensor>,
    kernel_gradients: Vec<Tensor>,
    input: Option<Tensor>,
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_diameter: usize) -> Self {
        assert!(in_channels > 0);
        assert!(kernel_diameter % 2 == 1);

        let kernels: Vec<Tensor> = (0..out_channels)
            .map(|_| Tensor::new_random(kernel_diameter, in_channels, 3))
            .collect();

        let kernel_gradients: Vec<Tensor> = (0..out_channels)
            .map(|_| Tensor::new(kernel_diameter, kernel_diameter, in_channels))
            .collect();

        Self {
            kernels,
            input: None,
            kernel_gradients,
        }
    }
}

impl Layer for Conv2D {
    fn forward(&mut self, tensor: Tensor) -> Tensor {
        todo!()
    }

    fn backward(&mut self, gradient: Tensor) -> Tensor {
        todo!()
    }

    fn update(&mut self, learning_rate: f32) {
        for (kernel, k_gradient) in self
            .kernels
            .iter_mut()
            .zip(self.kernel_gradients.iter_mut())
        {
            let d = kernel.width();
            let channels = kernel.channels();

            for ch in 0..channels {
                for y in 0..d {
                    for x in 0..d {
                        let w = kernel.get_xyz(x, y, ch);
                        let g = k_gradient.get_xyz(x, y, ch);
                        kernel.set_xyz(x, y, ch, w - learning_rate * g);
                    }
                }
            }

            *k_gradient = Tensor::new(d, d, channels);
        }
    }
}
