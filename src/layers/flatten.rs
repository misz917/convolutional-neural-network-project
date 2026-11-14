use crate::{layers::Layer, tensor::Tensor};

pub struct Flatten {
    original_dimensions: Option<(usize, usize, usize)>,
}

impl Flatten {
    pub fn new() -> Self {
        Flatten {
            original_dimensions: None,
        }
    }
}

impl Layer for Flatten {
    fn forward(&mut self, tensor: Tensor) -> Tensor {
        self.original_dimensions = Some(tensor.dimensions());

        let maps = tensor.get_maps();
        let merged = maps.concat().into_boxed_slice();

        Tensor::from(merged)
    }

    fn backward(&mut self, gradient: Tensor) -> Tensor {
        let (o_width, o_height, o_channels) = self
            .original_dimensions
            .expect("backward called before forward");
        let (g_width, g_height, g_channels) = gradient.dimensions();

        assert_eq!(
            o_width * o_height * o_channels,
            g_width * g_height * g_channels
        );

        let map = gradient.single_map();
        let reconstructed_maps: Vec<Box<[f32]>> = map
            .chunks_exact(o_width * o_height)
            .map(|chunk| chunk.to_vec().into_boxed_slice())
            .collect();

        Tensor::new_from_maps(reconstructed_maps, o_width, o_height)
    }

    fn update(&mut self, _learning_rate: f32) {
        // do nothing
    }
}
