pub struct Kernel {
    size: usize,
    weights: Vec<[f32; 3]>,
}

impl Kernel {
    pub fn new_random(size: usize) -> Self {
        let weights: Vec<[f32; 3]> = (0..size * size)
            .map(|_| {
                [
                    rand::random::<f32>() / 3.0,
                    rand::random::<f32>() / 3.0,
                    rand::random::<f32>() / 3.0,
                ]
            })
            .collect();
        Self { size, weights }
    }

    pub fn get_size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn get_xy(&self, x: usize, y: usize) -> [f32; 3] {
        self.weights[y * 3 + x]
    }

    #[inline]
    pub fn get_xyz(&self, x: usize, y: usize, z: usize) -> f32 {
        self.weights[y * 3 + x][z]
    }

    pub fn weights(&self) -> &[[f32; 3]] {
        &self.weights
    }
}
