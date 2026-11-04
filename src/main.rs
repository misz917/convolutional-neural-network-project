use image::ImageReader;

use crate::{cnn::CNN, pixels::Pixels};

mod cnn;
mod layers;
mod pixels;

fn main() {
    // tracing_subscriber::fmt().init();

    let image = image::open("image.png").unwrap();
    let pixels: Pixels = image.into();

    let cnn = CNN::new();
    // cnn.forward();
}
