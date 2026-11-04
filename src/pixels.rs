use image::DynamicImage;

pub struct Pixels(pub Vec<Vec<[u8; 3]>>);

impl From<DynamicImage> for Pixels {
    fn from(value: DynamicImage) -> Self {
        let image = value.to_rgb8();

        let width = image.width();

        let pixels: Vec<Vec<[u8; 3]>> = image
            .as_raw()
            .chunks_exact((width * 3).try_into().unwrap())
            .map(|row| row.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect())
            .collect();

        Self(pixels)
    }
}

impl Pixels {
    #[inline]
    pub fn xy(&self, x: usize, y: usize) -> &[u8; 3] {
        &self.0[x][y]
    }
}
