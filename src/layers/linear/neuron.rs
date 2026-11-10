pub struct Neuron {
    pub weights: Box<[f32]>,
    pub bias: f32,
    weights_gradient: Box<[f32]>,
    bias_gradient: f32,
}

impl Neuron {
    pub fn new_random(inputs: usize) -> Self {
        let bias = 0.0;
        let weights: Box<[f32]> = (0..inputs).map(|_| rand::random::<f32>() - 0.5).collect();

        Self {
            weights,
            bias,
            weights_gradient: vec![0.0; inputs].into(),
            bias_gradient: 0.0,
        }
    }

    pub fn forward(&mut self, input: &[f32]) -> f32 {
        assert_eq!(self.weights.len(), input.len());

        let total: f32 = self.weights.iter().zip(input).map(|(w, i)| w * i).sum();

        total + self.bias
    }

    // last_input passed as parameter because it's the same for all the neurons
    // last_input_gradient passed as mutable reference to avoid heap allocations just to return output
    pub fn backward(&mut self, gradient: f32, last_input: &[f32], last_input_gradient: &mut [f32]) {
        self.bias_gradient += gradient;

        for i in 0..self.weights.len() {
            self.weights_gradient[i] += gradient * last_input[i];
            last_input_gradient[i] += gradient * self.weights[i];
        }
    }

    pub fn update(&mut self, learning_rate: f32) {
        for (w, wg) in self
            .weights
            .iter_mut()
            .zip(self.weights_gradient.iter_mut())
        {
            *w -= *wg * learning_rate;
            *wg = 0.0;
        }

        self.bias -= self.bias_gradient * learning_rate;
        self.bias_gradient = 0.0;
    }
}
