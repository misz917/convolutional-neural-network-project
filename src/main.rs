use crate::{
    layers::{Layer, conv2d::Conv2D, maxpool2d::Maxpool2D, relu::ReLU},
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

    let t0 = Tensor::from(&image);
    let t1 = conv2d.apply(t0);
    let t2 = relu.apply(t1);
    let t3 = maxpool.apply(t2);
    println!("{:?}", t3);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv2d_relu_maxpool_test() {
        let mut t0 = Tensor::new(3, 3, 1);
        let t_maps = t0.get_maps_mut();
        let t_map = &mut t_maps[0];
        let designed_t_map: Box<[f32]> = Box::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        *t_map = designed_t_map;

        let mut conv2d = Conv2D::new(1, 1, 1);
        let kernels = conv2d.get_kernels_mut();
        let kernel = &mut kernels[0];
        let k_maps = kernel.get_maps_mut();
        let k_map = &mut k_maps[0];
        let designed_k_map: Box<[f32]> = Box::new([0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0]);
        *k_map = designed_k_map;

        let relu = ReLU;
        let maxpool = Maxpool2D;

        let t1 = conv2d.apply(t0);
        let t2 = relu.apply(t1);
        let t3 = maxpool.apply(t2);

        let mut expected_t3 = Tensor::new(1, 1, 1);
        let et3m = &mut expected_t3.get_maps_mut()[0];
        let designed_et3m: Box<[f32]> = Box::new([25.0]);
        *et3m = designed_et3m;

        assert_eq!(t3, expected_t3);
    }
}
