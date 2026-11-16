use std::ops::Range;

use image::{DynamicImage, GenericImage, GenericImageView, Rgba};

#[derive(Debug, PartialEq, Clone)]
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

    pub fn new_from_maps(maps: Vec<Box<[f32]>>, width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            maps,
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

        f32vec.into_boxed_slice()
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

    /// consumes other to append its maps to self
    pub fn merge(&mut self, other: Tensor) {
        self.maps.extend(other.maps);
    }

    pub fn get_maps(&self) -> &[Box<[f32]>] {
        &self.maps
    }

    pub fn single_map(&self) -> &Box<[f32]> {
        &self.maps[0]
    }

    /// uses 0 padding
    pub fn get_chunk(
        &self,
        center: (usize, usize),
        radius: usize,
        channel_range: Range<usize>,
    ) -> Option<Tensor> {
        let (this_width, this_height, this_channels) = self.dimensions();
        if center.0 >= this_width
            || center.1 >= this_height
            || channel_range.start >= this_channels
            || channel_range.end > this_channels
        {
            return None;
        }

        let r = radius as i32;
        let x = center.0 as i32;
        let y = center.1 as i32;
        let width = 1 + 2 * radius;

        let mut out = Self {
            width,
            height: width,
            maps: vec![vec![0.0; width * width].into_boxed_slice(); channel_range.len()],
        };

        for (out_ch, ch) in channel_range.enumerate() {
            for w in (x - r)..=(x + r) {
                for h in (y - r)..=(y + r) {
                    let value: f32;
                    if w < 0 || h < 0 || w >= this_width as i32 || h >= this_height as i32 {
                        value = 0.0;
                    } else {
                        value = self.get_xyz(w as usize, h as usize, ch);
                    }

                    let local_x = (w - x + r) as usize;
                    let local_y = (h - y + r) as usize;
                    out.set_xyz(local_x, local_y, out_ch, value);
                }
            }
        }

        Some(out)
    }

    pub fn add_chunk(&mut self, center: (usize, usize), chunk: &Tensor, scale: f32) {
        assert_eq!(chunk.channels(), self.channels());
        let r = (chunk.width() / 2) as i32;

        for ch in 0..chunk.channels() {
            for yi in 0..chunk.height() {
                for xi in 0..chunk.width() {
                    let x = center.0 as i32 + xi as i32 - r;
                    let y = center.1 as i32 + yi as i32 - r;
                    if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
                        continue;
                    }

                    let (x, y) = (x as usize, y as usize);
                    let current = self.get_xyz(x, y, ch);
                    let added = scale * chunk.get_xyz(xi, yi, ch);
                    self.set_xyz(x, y, ch, current + added);
                }
            }
        }
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

                for (channel, p) in pixel.iter().enumerate() {
                    tensor.set_xyz(
                        x.try_into().unwrap(),
                        y.try_into().unwrap(),
                        channel,
                        (*p as f32) / 255.0,
                    );
                }
            }
        }

        tensor
    }
}

// TODO: improve for cases when there are more than 3 channels
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
