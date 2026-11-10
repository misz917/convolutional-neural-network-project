use crate::{
    cnn::CNN,
    layers::{conv2d::Conv2D, flatten::Flatten, linear::Linear, maxpool2d::Maxpool2D, relu::ReLU},
    tensor::Tensor,
};

mod cnn;
pub mod kernel;
mod layers;
pub mod tensor;

fn main() {
    let mut cnn = CNN::new(vec![
        Box::new(Conv2D::new(3, 16, 3)),
        Box::new(ReLU),
        Box::new(Maxpool2D),
        Box::new(Flatten),
        Box::new(Linear::new(96, 5)),
    ]);

    let image = image::open("colors.png").unwrap();
    let results = cnn.forward(image);
    println!("{:?}", results);
}

fn mse_gradient(prediction: &[f32], target: &[f32]) -> Tensor {
    assert_eq!(prediction.len(), target.len());
    assert!(prediction.len() > 0);

    let n = prediction.len() as f32;

    let gradient: Box<[f32]> = prediction
        .iter()
        .zip(target)
        .map(|(p, t)| 2.0 * (p - t) / n)
        .collect();

    Tensor::from(gradient)
}
