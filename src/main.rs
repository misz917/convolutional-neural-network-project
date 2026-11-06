use image::DynamicImage;

use crate::{
    cnn::CNN,
    layers::{conv2d::Conv2D, relu::ReLU},
    tensor::Tensor,
};

mod cnn;
pub mod kernel;
mod layers;
pub mod tensor;

fn main() {
    let image = image::open("colors.png").unwrap();

    let conv2d = Conv2D::new(3, 1, 1);
    let tensor1 = Tensor::from(&image);
    let mut tensor2 = conv2d.apply(&tensor1);
    ReLU::apply(&mut tensor2);
    println!("{:?}", tensor2);
}
