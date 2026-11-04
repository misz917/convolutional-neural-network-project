use image::DynamicImage;

pub struct Tensor {
    width: usize,
    height: usize,
    channels: usize,
    maps: Vec<Box<[f32]>>,
}

impl Tensor {
    pub fn new() -> Self {
        todo!()
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

impl From<DynamicImage> for Tensor {
    fn from(value: DynamicImage) -> Self {
        todo!()
    }
}
