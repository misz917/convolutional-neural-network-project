use image::{DynamicImage, GenericImage, GenericImageView, Rgba};

#[derive(Debug, PartialEq)]
pub struct Tensor {
    width: usize,
    height: usize,
    maps: Vec<Box<[f32]>>,
}

impl Tensor {
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        Self {
            width,
            height,
            maps: vec![vec![0.0; width * height].into_boxed_slice(); channels],
        }
    }

    /// worse version of Kaiming Initialization
    pub fn new_random(width: usize, height: usize, channels: usize) -> Self {
        let fan_in = (width * height * channels) as f32;
        let limit = f32::sqrt(2.0 / fan_in);

        let maps: Vec<Box<[f32]>> = (0..channels)
            .map(|_| Self::new_random_channel(width, height, limit))
            .collect();

        Tensor {
            width,
            height,
            maps,
        }
    }

    fn new_random_channel(width: usize, height: usize, limit: f32) -> Box<[f32]> {
        let wh = width * height;
        let f32vec: Vec<f32> = (0..wh).map(|_| rand::random_range(-limit..limit)).collect();
        let f32box = f32vec.into_boxed_slice();

        f32box
    }

    /// returns (width, height, channels)
    #[inline]
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width(), self.height(), self.channels())
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
        self.maps.len()
    }

    pub fn merge(&mut self, other: Tensor) {
        self.maps.extend(other.maps);
    }

    /// very unsafe to use
    pub fn get_maps_mut(&mut self) -> &mut [Box<[f32]>] {
        &mut self.maps
    }

    pub fn get_maps(&self) -> &[Box<[f32]>] {
        &self.maps
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

impl From<Box<[f32]>> for Tensor {
    fn from(value: Box<[f32]>) -> Self {
        Tensor {
            width: value.len(),
            height: 1,
            maps: vec![value],
        }
    }
}
