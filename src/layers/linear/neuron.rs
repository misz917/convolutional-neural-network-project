pub struct Neuron {
    pub weights: Box<[f32]>,
    pub bias: f32,
}

impl Neuron {
    pub fn new_random(inputs: usize) -> Self {
        let bias = 0.0;
        let weights: Box<[f32]> = (0..inputs).map(|_| rand::random::<f32>() - 0.5).collect();

        Self { weights, bias }
    }

    pub fn apply(&self, input: &[f32]) -> f32 {
        assert_eq!(self.weights.len(), input.len());

        let total: f32 = self.weights.iter().zip(input).map(|(w, i)| w * i).sum();

        total + self.bias
    }
}
