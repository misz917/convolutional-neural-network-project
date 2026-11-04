use image::DynamicImage;

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
        let filtered_image = image.filter3x3(kernel.weights());
        filtered_image
    }
}
