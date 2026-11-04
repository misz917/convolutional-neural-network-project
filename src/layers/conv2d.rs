use image::{DynamicImage, GenericImageView, Rgba, RgbaImage};

use crate::layers::Kernel;

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

    pub fn apply(&self, image: DynamicImage) -> Vec<DynamicImage> {
        let output: Vec<DynamicImage> = self
            .kernels
            .iter()
            .map(|k| Self::apply_filter(&image, k))
            .collect();

        output
    }

    fn apply_filter(image: &DynamicImage, kernel: &Kernel) -> DynamicImage {
        let (width, height) = image.dimensions();
        let mut new_image = DynamicImage::new_rgb8(width, height);

        for w in 0..width {
            for h in 0..height {
                //
            }
        }

        todo!()
    }
}
