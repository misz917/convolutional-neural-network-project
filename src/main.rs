use crate::{
    layers::{
        Layer, conv2d::Conv2D, flatten::Flatten, linear::Linear, maxpool2d::Maxpool2D, relu::ReLU,
    },
    tensor::Tensor,
};

mod cnn;
pub mod kernel;
mod layers;
pub mod tensor;

fn main() {
    let image = image::open("colors.png").unwrap();

    let conv2d = Conv2D::new(3, 1, 1);
    let relu = ReLU;
    let maxpool = Maxpool2D;
    let flatten = Flatten;
    let linear = Linear::new(6, 2);

    let t0 = Tensor::from(&image);
    let t1 = conv2d.apply(t0);
    let t2 = relu.apply(t1);
    let t3 = maxpool.apply(t2);
    let t4 = flatten.apply(t3);
    let t5 = linear.apply(t4);

    println!("{:?}", t5);
}
