use crate::{
    cnn::CNN,
    layers::{conv2d::Conv2D, flatten::Flatten, linear::Linear, maxpool2d::Maxpool2D, relu::ReLU},
};

mod cnn;
pub mod kernel;
mod layers;
pub mod tensor;

fn main() {
    let cnn = CNN::new(vec![
        Box::new(Conv2D::new(3, 1, 1)),
        Box::new(ReLU),
        Box::new(Maxpool2D),
        Box::new(Flatten),
        Box::new(Linear::new(6, 2)),
    ]);

    let image = image::open("colors.png").unwrap();
    let results = cnn.forward(image);
    println!("{:?}", results);
}
