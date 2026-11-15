use crate::{layers::Layer, tensor::Tensor};

pub struct Conv2D {
    kernels: Vec<Tensor>,
    kernel_diameter: usize,
    kernel_gradients: Vec<Tensor>,
    input: Option<Tensor>,
}

impl Conv2D {
    pub fn new(in_channels: usize, out_channels: usize, kernel_diameter: usize) -> Self {
        assert!(in_channels > 0);
        assert!(kernel_diameter % 2 == 1);

        let kernels: Vec<Tensor> = (0..out_channels)
            .map(|_| Tensor::new_random(kernel_diameter, kernel_diameter, in_channels))
            .collect();

        let kernel_gradients: Vec<Tensor> = (0..out_channels)
            .map(|_| Tensor::new(kernel_diameter, kernel_diameter, in_channels))
            .collect();

        Self {
            kernels,
            input: None,
            kernel_gradients,
            kernel_diameter,
        }
    }
}

impl Layer for Conv2D {
    fn forward(&mut self, tensor: Tensor) -> Tensor {
        let (width, height, channels) = tensor.dimensions();
        assert_eq!(channels, self.kernels[0].channels());

        let mut out = Tensor::new(width, height, self.kernels.len());

        for (c, kernel) in self.kernels.iter().enumerate() {
            for w in 0..width {
                for h in 0..height {
                    let Some(chunk) =
                        tensor.get_chunk((w, h), self.kernel_diameter / 2, 0..channels)
                    else {
                        continue;
                    };

                    let multiplied = multiply_tensors(kernel, &chunk);
                    let sum = sum_tensor(&multiplied);
                    out.set_xyz(w, h, c, sum);
                }
            }
        }

        self.input = Some(tensor);
        out
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

fn multiply_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.dimensions(), b.dimensions());
    let (width, height, channels) = a.dimensions();

    let mut out = Tensor::new(width, height, channels);

    for c in 0..channels {
        for x in 0..width {
            for y in 0..height {
                let a_val = a.get_xyz(x, y, c);
                let b_val = b.get_xyz(x, y, c);
                out.set_xyz(x, y, c, a_val * b_val);
            }
        }
    }

    out
}

fn sum_tensor(tensor: &Tensor) -> f32 {
    let (width, height, channels) = tensor.dimensions();

    let mut sum = 0.0;

    for c in 0..channels {
        for x in 0..width {
            for y in 0..height {
                sum += tensor.get_xyz(x, y, c);
            }
        }
    }

    sum
}
