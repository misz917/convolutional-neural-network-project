use image::{DynamicImage, GenericImage, Rgba};

pub struct Tensor {
    width: usize,
    height: usize,
    data: Box<[f32]>,
}

impl Tensor {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height].into_boxed_slice(),
        }
    }

    #[inline]
    pub fn put_xy(&mut self, x: usize, y: usize, value: f32) {
        self.data[y * self.width + x] = value;
    }

    #[inline]
    pub fn get_xy(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }
}

// TODO: Change to From<&Tensor> for DynamicImage
impl Into<DynamicImage> for &Tensor {
    fn into(self) -> DynamicImage {
        let mut new_image = DynamicImage::new_rgb8(
            self.width.try_into().unwrap(),
            self.height.try_into().unwrap(),
        );

        for x in 0..self.width {
            for y in 0..self.height {
                let pixel = (self.get_xy(x, y) * 255.0).clamp(0.0, 255.0) as u8;
                let rgba: Rgba<u8> = Rgba([pixel; 4]);
                new_image.put_pixel(x as u32, y as u32, rgba);
            }
        }

        new_image
    }
}
