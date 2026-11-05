use image::DynamicImage;

use crate::{cnn::CNN, layers::conv2d::Conv2D, tensor::Tensor};

mod cnn;
pub mod kernel;
mod layers;
pub mod tensor;

fn main() {
    // tracing_subscriber::fmt().init();

    // let image = image::open("lenna.jpg").unwrap();
    let image = image::open("colors.png").unwrap();

    let conv2d = Conv2D::new(1, 1, 3);
    let input_tensor = Tensor::from(&image);
    let output_tensor = conv2d.apply(&input_tensor);
    println!("{:?}", output_tensor);

    // for (i, t) in tensors.iter().enumerate() {
    //     let img: DynamicImage = t.into();
    //     img.save(format!("./{}.png", i)).unwrap();
    // }

    // let tensor = Tensor::from(&image);
    // println!("{:?}", tensor);
    // let r_image = DynamicImage::from(&tensor);
    // r_image.save("r_image.png").unwrap();

    // let cnn = CNN::new();
    // cnn.forward();
}
