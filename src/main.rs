use std::collections::HashMap;

use clap::Parser;

use crate::{
    cli::Cli,
    cnn::CNN,
    data_loader::DataLoader,
    layers::{conv2d::Conv2D, flatten::Flatten, linear::Linear, maxpool2d::Maxpool2D, relu::ReLU},
    trainer::Trainer,
};

mod cli;
mod cnn;
mod data_loader;
mod layers;
pub mod tensor;
mod trainer;

fn main() {
    let cli = Cli::parse();
    tracing_subscriber::fmt()
        .with_max_level(cli.log_level)
        .init();

    let data_loader = DataLoader::new("cifar-10-png/train".into());
    let targets = data_loader.get_targets().clone();
    let mut cnn = CNN::new(vec![
        Box::new(Conv2D::new(3, 8, 3)),
        Box::new(ReLU::new()),
        Box::new(Maxpool2D::new()),
        Box::new(Flatten::new()),
        Box::new(Linear::new(2048, data_loader.num_classes())),
    ]);

    let mut trainer = Trainer::new(&mut cnn, data_loader, (32, 32), cli.epochs, 0.01);
    trainer.run();

    let bird1 = image::open("cifar-10-png/test/bird/0004.png").unwrap();
    let bird2 = image::open("cifar-10-png/test/bird/0010.png").unwrap();
    let truck = image::open("cifar-10-png/test/truck/0003.png").unwrap();
    let cat = image::open("cifar-10-png/test/cat/0006.png").unwrap();

    println!("{:?}", labels_in_order(&targets));

    let bird1_prediction = cnn.forward(bird1);
    println!("{:?}", bird1_prediction);

    let bird2_prediction = cnn.forward(bird2);
    println!("{:?}", bird2_prediction);

    let truck_prediction = cnn.forward(truck);
    println!("{:?}", truck_prediction);

    let cat_prediction = cnn.forward(cat);
    println!("{:?}", cat_prediction);
}

fn labels_in_order(map: &HashMap<String, Vec<f32>>) -> Vec<String> {
    let mut labels = vec![String::new(); map.len()];

    for (name, one) in map {
        let index = one.iter().position(|&v| v == 1.0).unwrap();
        labels[index] = name.clone();
    }

    labels
}
