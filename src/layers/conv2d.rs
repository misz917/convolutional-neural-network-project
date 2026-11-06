use crate::{kernel::Kernel, tensor::Tensor};

pub struct Conv2D {
    kernels: Vec<Kernel>,
}

impl Conv2D {
    pub fn new(kernel_num: usize, kernel_radius: usize, kernel_height: usize) -> Self {
        assert!(kernel_height > 0);
        let kernels: Vec<Kernel> = (0..kernel_num)
            .map(|_| Kernel::new_random(kernel_radius, kernel_height))
            .collect();

        Self { kernels }
    }

    pub fn apply(&self, tensor: &Tensor) -> Tensor {
        let mut output_tensor = Tensor::new(tensor.width(), tensor.height(), 0);

        for kernel in &self.kernels {
            let other = Self::convolve(tensor, kernel).unwrap();
            output_tensor.merge(other);
        }

        output_tensor
    }

    fn convolve(tensor: &Tensor, kernel: &Kernel) -> Result<Tensor, String> {
        let t_channels = tensor.channels();
        let channels = kernel.channels();
        if t_channels != channels {
            return Err(format!(
                "Tensor channels != Kernel channels ({}!={})",
                t_channels, channels
            )
            .to_owned());
        }

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

        Ok(new_tensor)
    }
}
