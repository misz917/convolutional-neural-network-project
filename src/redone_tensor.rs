use image::{DynamicImage, GenericImage, GenericImageView, Rgba};

#[derive(Debug)]
pub struct Tensor {
    width: usize,
    height: usize,
    channels: usize,
    maps: Vec<Box<[f32]>>,
}

impl Tensor {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            width,
            height,
            channels,
            maps: vec![vec![0.0; width * height].into_boxed_slice(); 3],
        }
    }

    #[inline]
    pub fn get_xyz(&self, x: usize, y: usize, channel: usize) -> f32 {
        self.maps[channel][y * self.width + x]
    }

    #[inline]
    pub fn set_xyz(&mut self, x: usize, y: usize, channel: usize, value: f32) {
        self.maps[channel][y * self.width + x] = value;
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn channels(&self) -> usize {
        self.channels
    }
}

impl From<&DynamicImage> for Tensor {
    fn from(value: &DynamicImage) -> Self {
        let (width, height) = value.dimensions();
        let rgb = value.to_rgb8();
        let mut tensor = Tensor::new(width.try_into().unwrap(), height.try_into().unwrap(), 3);

        for x in 0..width {
            for y in 0..height {
                let pixel = rgb.get_pixel(x, y).0;

                for channel in 0..3 {
                    tensor.set_xyz(
                        x.try_into().unwrap(),
                        y.try_into().unwrap(),
                        channel,
                        (pixel[channel] as f32) / 255.0,
                    );
                }
            }
        }

        tensor
    }
}

// TODO: improve for cases when there are more than 3 channel
impl From<&Tensor> for DynamicImage {
    fn from(value: &Tensor) -> Self {
        let width = value.width();
        let height = value.height();
        let cnum = value.channels();

        let mut image =
            DynamicImage::new_rgb8(width.try_into().unwrap(), height.try_into().unwrap());

        for x in 0..width {
            for y in 0..height {
                let r = (value.get_xyz(x, y, 0 % cnum) * 255.0).clamp(0.0, 255.0) as u8;
                let g = (value.get_xyz(x, y, 1 % cnum) * 255.0).clamp(0.0, 255.0) as u8;
                let b = (value.get_xyz(x, y, 2 % cnum) * 255.0).clamp(0.0, 255.0) as u8;
                image.put_pixel(
                    x.try_into().unwrap(),
                    y.try_into().unwrap(),
                    Rgba([r, g, b, 1]),
                );
            }
        }

        image
    }
}
