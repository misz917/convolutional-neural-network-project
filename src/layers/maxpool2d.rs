use crate::{layers::Layer, tensor::Tensor};

pub struct Maxpool2D;

impl Layer for Maxpool2D {
    fn forward(&mut self, tensor: Tensor) -> Tensor {
        let (width, height, channels) = tensor.dimensions();

        let mut output = Tensor::new(width / 2, height / 2, channels);

        for c in 0..channels {
            for y in (1..height).step_by(2) {
                for x in (1..width).step_by(2) {
                    let ul = tensor.get_xyz(x - 1, y - 1, c);
                    let ur = tensor.get_xyz(x, y - 1, c);
                    let dl = tensor.get_xyz(x - 1, y, c);
                    let dr = tensor.get_xyz(x, y, c);

                    let max_val = ul.max(ur).max(dl).max(dr);
                    output.set_xyz(x / 2, y / 2, c, max_val);
                }
            }
        }

        output
    }

    fn backward(&mut self, gradient: Tensor) -> Tensor {
        todo!()
    }

    fn update(&mut self, learning_rate: f32) {
        todo!()
    }
}
