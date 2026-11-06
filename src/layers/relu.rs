use crate::tensor::Tensor;

pub struct ReLU;

impl ReLU {
    pub fn apply(tensor: &mut Tensor) {
        let width = tensor.width();
        let height = tensor.height();
        let channels = tensor.channels();

        for c in 0..channels {
            for x in 0..width {
                for y in 0..height {
                    if tensor.get_xyz(x, y, c) < 0.0 {
                        tensor.set_xyz(x, y, c, 0.0);
                    }
                }
            }
        }
    }
}
