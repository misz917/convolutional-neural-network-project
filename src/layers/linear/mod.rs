use crate::{
    layers::{Layer, linear::neuron::Neuron},
    tensor::Tensor,
};

mod neuron;

pub struct Linear {
    neurons: Box<[Neuron]>,
    last_input: Option<Tensor>,
}

impl Layer for Linear {
    fn forward(&mut self, tensor: Tensor) -> Tensor {
        assert_eq!(tensor.height(), 1);
        assert_eq!(tensor.channels(), 1);
        assert_eq!(tensor.width(), self.neurons[0].weights.len());

        let input = &tensor.get_maps()[0];
        let mut output: Vec<f32> = Vec::new();

        for n in &mut self.neurons {
            let value = n.forward(input);
            output.push(value);
        }

        self.last_input = Some(tensor);
        Tensor::from(output.into_boxed_slice())
    }

    fn backward(&mut self, gradient: Tensor) -> Tensor {
        let last_input = self.last_input.as_ref().unwrap().single_map();
        let gradient = gradient.single_map();
        let mut last_input_gradient = vec![0.0; last_input.len()];

        for (neuron, g) in self.neurons.iter_mut().zip(gradient) {
            neuron.backward(*g, last_input, &mut last_input_gradient);
        }

        Tensor::from(last_input_gradient.into_boxed_slice())
    }

    fn update(&mut self, learning_rate: f32) {
        for neuron in &mut self.neurons {
            neuron.update(learning_rate);
        }
    }
}

impl Linear {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        assert!(inputs > 0);
        assert!(outputs > 0);

        let neurons: Box<[Neuron]> = (0..outputs).map(|_| Neuron::new_random(inputs)).collect();

        Self {
            neurons,
            last_input: None,
        }
    }
}
