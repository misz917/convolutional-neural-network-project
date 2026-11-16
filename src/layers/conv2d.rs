use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

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

        let mut smaller_tensors: Vec<(Tensor, usize)> = self
            .kernels
            .par_iter()
            .enumerate()
            .map(|(c, kernel)| {
                let mut out = Tensor::new(width, height, 1);

                for w in 0..width {
                    for h in 0..height {
                        let Some(chunk) =
                            tensor.get_chunk((w, h), self.kernel_diameter / 2, 0..channels)
                        else {
                            continue;
                        };

                        let multiplied = multiply_tensors(kernel, &chunk);
                        let sum = sum_tensor(&multiplied);
                        out.set_xyz(w, h, 0, sum);
                    }
                }

                (out, c)
            })
            .collect();

        smaller_tensors.sort_by(|(_, ai), (_, bi)| ai.cmp(bi));
        let mut merged_tensor = Tensor::new(width, height, 0);
        for (t, _) in smaller_tensors {
            merged_tensor.merge(t);
        }

        self.input = Some(tensor);
        merged_tensor
    }

    fn backward(&mut self, gradient: Tensor) -> Tensor {
        let input = self.input.as_ref().expect("backward called before forward");
        let (width, height, in_channels) = input.dimensions();
        assert_eq!(gradient.dimensions(), (width, height, self.kernels.len()));

        let mut out = Tensor::new(width, height, in_channels);

        for w in 0..width {
            for h in 0..height {
                let chunk = input
                    .get_chunk((w, h), self.kernel_diameter / 2, 0..in_channels)
                    .unwrap();

                for (ch, (kernel, k_gradient)) in self
                    .kernels
                    .iter()
                    .zip(self.kernel_gradients.iter_mut())
                    .enumerate()
                {
                    let g = gradient.get_xyz(w, h, ch);

                    add_scaled(k_gradient, &chunk, g);
                    out.add_chunk((w, h), kernel, g);
                }
            }
        }

        out
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

/// target += scale * source
fn add_scaled(target: &mut Tensor, source: &Tensor, scale: f32) {
    assert_eq!(target.dimensions(), source.dimensions());
    let (width, height, channels) = target.dimensions();

    for c in 0..channels {
        for x in 0..width {
            for y in 0..height {
                let value = target.get_xyz(x, y, c) + scale * source.get_xyz(x, y, c);
                target.set_xyz(x, y, c, value);
            }
        }
    }
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
