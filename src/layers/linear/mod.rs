use crate::{
    layers::{Layer, linear::neuron::Neuron},
    tensor::Tensor,
};

mod neuron;

pub struct Linear {
    neurons: Box<[Neuron]>,
}

impl Layer for Linear {
    fn apply(&mut self, tensor: Tensor) -> Tensor {
        assert_eq!(tensor.height(), 1);
        assert_eq!(tensor.channels(), 1);
        assert_eq!(tensor.width(), self.neurons[0].weights.len());

        let input = &tensor.get_maps()[0];
        let mut output: Vec<f32> = Vec::new();

        for n in &self.neurons {
            let value = n.apply(input);
            output.push(value);
        }

        Tensor::from(output.into_boxed_slice())
    }
}

impl Linear {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        assert!(inputs > 0);
        assert!(outputs > 0);

        let neurons: Box<[Neuron]> = (0..outputs).map(|_| Neuron::new_random(inputs)).collect();

        Self { neurons }
    }
}
