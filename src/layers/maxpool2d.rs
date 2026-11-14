use crate::{layers::Layer, tensor::Tensor};

pub struct Maxpool2D {
    original_dimensions: Option<(usize, usize, usize)>,
    max_positions: Option<Tensor>,
}

impl Maxpool2D {
    pub fn new() -> Self {
        Maxpool2D {
            original_dimensions: None,
            max_positions: None,
        }
    }
}

impl Layer for Maxpool2D {
    fn forward(&mut self, tensor: Tensor) -> Tensor {
        self.original_dimensions = Some(tensor.dimensions());
        let (width, height, channels) = tensor.dimensions();

        let mut output = Tensor::new(width / 2, height / 2, channels);
        let mut max_positions = output.clone();

        for c in 0..channels {
            for y in (1..height).step_by(2) {
                for x in (1..width).step_by(2) {
                    let ul = tensor.get_xyz(x - 1, y - 1, c);
                    let ur = tensor.get_xyz(x, y - 1, c);
                    let dl = tensor.get_xyz(x - 1, y, c);
                    let dr = tensor.get_xyz(x, y, c);

                    let max_val = ul.max(ur).max(dl).max(dr);
                    output.set_xyz(x / 2, y / 2, c, max_val);

                    let pos = if max_val == ul {
                        0
                    } else if max_val == ur {
                        1
                    } else if max_val == dl {
                        2
                    } else {
                        3
                    };
                    max_positions.set_xyz(x / 2, y / 2, c, pos as f32);
                }
            }
        }

        self.max_positions = Some(max_positions);
        output
    }

    fn backward(&mut self, gradient: Tensor) -> Tensor {
        let (width, height, channels) = self
            .original_dimensions
            .expect("backward called before forward");
        let max_positions = self.max_positions.as_mut().unwrap();

        let mut inputsize_gradient = Tensor::new(width, height, channels);

        for c in 0..channels {
            for oy in 0..height / 2 {
                for ox in 0..width / 2 {
                    let pos = max_positions.get_xyz(ox, oy, c) as usize;
                    let (dx, dy) = (pos % 2, pos / 2); // 0=ul, 1=ur, 2=dl, 3=dr
                    inputsize_gradient.set_xyz(
                        ox * 2 + dx,
                        oy * 2 + dy,
                        c,
                        gradient.get_xyz(ox, oy, c),
                    );
                }
            }
        }

        inputsize_gradient
    }

    fn update(&mut self, _learning_rate: f32) {
        // do nothing
    }
}
