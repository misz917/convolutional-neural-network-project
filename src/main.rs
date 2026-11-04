use crate::cnn::CNN;

mod cnn;
pub mod kernel;
mod layers;
pub mod tensor;

fn main() {
    // tracing_subscriber::fmt().init();

    let image = image::open("image.png").unwrap();

    let cnn = CNN::new();
    // cnn.forward();
}
