use crate::{kernel::Kernel, layers::Layer, tensor::Tensor};

pub struct Conv2D {
    kernels: Vec<Kernel>,
    kernel_gradients: Vec<Tensor>,
    input: Option<Tensor>,
}

impl Layer for Conv2D {
    fn forward(&mut self, tensor: Tensor) -> Tensor {
        let mut output_tensor = Tensor::new(tensor.width(), tensor.height(), 0);

        for kernel in &self.kernels {
            let other = Self::convolve(&tensor, kernel);
            output_tensor.merge(other);
        }

        self.input = Some(tensor);
        output_tensor
    }

    fn backward(&mut self, gradient: Tensor) -> Tensor {
        let input = self.input.as_ref().expect("backward called before forward");

        let width = input.width() as i32;
        let height = input.height() as i32;
        let channels = input.channels();

        let mut input_grad = Tensor::new(width as usize, height as usize, channels);

        for (k, (kernel, kgrad)) in self
            .kernels
            .iter()
            .zip(self.kernel_gradients.iter_mut())
            .enumerate()
        {
            let k_diameter = kernel.diameter() as i32;
            let k_radius = k_diameter / 2;

            for tensor_y in 0..height {
                for tensor_x in 0..width {
                    let g = gradient.get_xyz(tensor_x as usize, tensor_y as usize, k);

                    for kernel_y in 0..k_diameter {
                        for kernel_x in 0..k_diameter {
                            let rx = (tensor_x + kernel_x - k_radius).clamp(0, width - 1) as usize;
                            let ry = (tensor_y + kernel_y - k_radius).clamp(0, height - 1) as usize;
                            let (kx, ky) = (kernel_x as usize, kernel_y as usize);

                            for ch in 0..channels {
                                let t_val = input.get_xyz(rx, ry, ch);
                                let k_val = kernel.get_xyz(kx, ky, ch);

                                let kg = kgrad.get_xyz(kx, ky, ch);
                                kgrad.set_xyz(kx, ky, ch, kg + g * t_val);

                                let ig = input_grad.get_xyz(rx, ry, ch);
                                input_grad.set_xyz(rx, ry, ch, ig + g * k_val);
                            }
                        }
                    }
                }
            }
        }

        input_grad
    }

    fn update(&mut self, learning_rate: f32) {
        for (kernel, k_gradient) in self
            .kernels
            .iter_mut()
            .zip(self.kernel_gradients.iter_mut())
        {
            let d = kernel.diameter();
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

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_diameter: usize) -> Self {
        assert!(in_channels > 0);
        assert!(kernel_diameter % 2 == 1);

        let kernels: Vec<Kernel> = (0..out_channels)
            .map(|_| Kernel::new_random(kernel_diameter, in_channels))
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

    fn convolve(tensor: &Tensor, kernel: &Kernel) -> Tensor {
        let t_channels = tensor.channels();
        let channels = kernel.channels();
        assert_eq!(t_channels, channels);

        let width = tensor.width() as i32;
        let height = tensor.height() as i32;
        let mut new_tensor = Tensor::new(width as usize, height as usize, 1);

        let k_diameter = kernel.diameter() as i32;
        let k_radius = k_diameter / 2;

        // for each pixel in tensor
        for tensor_x in 0..width {
            for tensor_y in 0..height {
                let mut sum = 0.0;
                // for each pixel in kernel
                for kernel_x in 0..k_diameter {
                    for kernel_y in 0..k_diameter {
                        // relative to the selected tensor pixel
                        // clamped to stay within bounds of the tensor
                        // TODO: change to 0 padding
                        let rx = (tensor_x + kernel_x - k_radius).clamp(0, width - 1);
                        let ry = (tensor_y + kernel_y - k_radius).clamp(0, height - 1);

                        for ch in 0..channels {
                            let t_val =
                                tensor.get_xyz(rx.try_into().unwrap(), ry.try_into().unwrap(), ch);
                            let k_val = kernel.get_xyz(
                                kernel_x.try_into().unwrap(),
                                kernel_y.try_into().unwrap(),
                                ch,
                            );

                            sum += t_val * k_val;
                        }
                    }
                }

                new_tensor.set_xyz(tensor_x as usize, tensor_y as usize, 0, sum);
            }
        }

        new_tensor
    }
}
