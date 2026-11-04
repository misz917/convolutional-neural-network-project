use image::DynamicImage;

use crate::{kernel::Kernel, tensor::Tensor};

pub struct Conv2D {
    kernels: Vec<Kernel>,
}

impl Conv2D {
    pub fn new(kernel_num: usize, kernel_size: usize) -> Self {
        assert!(kernel_size == 3);

        let kernels: Vec<Kernel> = (0..kernel_num)
            .map(|_| Kernel::new_random(kernel_size))
            .collect();

        Self { kernels }
    }

    pub fn apply(&self, image: DynamicImage) -> Vec<Tensor> {
        let output: Vec<Tensor> = self
            .kernels
            .iter()
            .map(|k| Self::convolve(&image, k))
            .collect();

        output
    }

    fn convolve(image: &DynamicImage, kernel: &Kernel) -> Tensor {
        let rgb_image = image.to_rgb8(); // convert once
        let (width, height) = rgb_image.dimensions();
        let (w, h) = (width as i32, height as i32);

        let k = kernel.get_size() as i32;
        let r = k / 2;

        let mut out = Tensor::new(width as usize, height as usize);

        for y in 0..h {
            for x in 0..w {
                let mut sum = 0.0;

                for ky in 0..k {
                    for kx in 0..k {
                        let ix = (x + kx - r).clamp(0, w - 1);
                        let iy = (y + ky - r).clamp(0, h - 1);

                        let pixel = rgb_image.get_pixel(ix as u32, iy as u32);
                        let kernel_val = kernel.get_xy(kx as usize, ky as usize);

                        sum += pixel[0] as f32 * kernel_val[0]
                            + pixel[1] as f32 * kernel_val[1]
                            + pixel[2] as f32 * kernel_val[2];
                    }
                }

                out.put_xy(x as usize, y as usize, sum);
            }
        }

        out
    }
}
