pub mod conv2d;
pub mod maxpool2d;
pub mod relu;

pub struct Tensor(Vec<Vec<u8>>);

struct Kernel {
    size: usize,
    weights: Vec<f32>,
}

impl Kernel {
    pub fn new_random(size: usize) -> Self {
        let weights: Vec<f32> = (0..size * size).map(|_| rand::random::<f32>()).collect();
        Self { size, weights }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn xy(&self, x: usize, y: usize) -> f32 {
        self.weights[y * 3 + x]
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
}
