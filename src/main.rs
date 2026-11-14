use clap::Parser;

use crate::{
    cli::Cli,
    cnn::CNN,
    data_loader::DataLoader,
    layers::{conv2d::Conv2D, flatten::Flatten, linear::Linear, maxpool2d::Maxpool2D, relu::ReLU},
    tensor::Tensor,
};

mod cli;
mod cnn;
mod data_loader;
pub mod kernel;
mod layers;
pub mod tensor;
mod trainer;

fn main() {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_max_level(cli.log_level)
        .init();

    let mut data_loader = DataLoader::new("cifar-10-png/train".into());
    let mut cnn = CNN::new(vec![
        Box::new(Conv2D::new(3, 4, 3)),
        Box::new(ReLU::new()),
        Box::new(Maxpool2D::new()),
        Box::new(Flatten::new()),
        Box::new(Linear::new(96, 5)),
    ]);

    let image = data_loader.next().unwrap().0;

    for i in 0..cli.epochs {
        println!("\nLoop {i}");
        let prediction = cnn.forward(image.clone());
        println!("{:?}", prediction);
        let gradient = mse_gradient(&prediction, &[0.0, 1.0, 0.0, 0.0, 0.0]);
        println!("{:?}", gradient);
        cnn.backward(gradient);
        cnn.update(0.01);
    }
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
