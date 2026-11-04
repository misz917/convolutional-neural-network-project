use image::DynamicImage;

use crate::{cnn::CNN, layers::conv2d::Conv2D};

mod cnn;
pub mod kernel;
mod layers;
mod redone_tensor;
pub mod tensor;

fn main() {
    // tracing_subscriber::fmt().init();

    let image = image::open("lenna.jpg").unwrap();

    let conv2d = Conv2D::new(2, 3);
    let tensors = conv2d.apply(image);

    for (i, t) in tensors.iter().enumerate() {
        let img: DynamicImage = t.into();
        img.save(format!("./{}.png", i)).unwrap();
    }

    // let cnn = CNN::new();
    // cnn.forward();
}
