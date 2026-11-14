use clap::Parser;

use crate::{
    cli::Cli,
    cnn::CNN,
    data_loader::DataLoader,
    layers::{conv2d::Conv2D, flatten::Flatten, linear::Linear, maxpool2d::Maxpool2D, relu::ReLU},
    tensor::Tensor,
    trainer::Trainer,
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
        Box::new(Linear::new(1024, data_loader.num_classes())),
    ]);

    let mut trainer = Trainer::new(&mut cnn, data_loader, (32, 32), 1, 0.001);
    trainer.run();
}
