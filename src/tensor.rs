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
}
