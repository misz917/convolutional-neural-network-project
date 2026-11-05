use crate::{kernel::Kernel, tensor::Tensor};

pub struct Conv2D {
    kernels: Vec<Kernel>,
}

impl Conv2D {
    pub fn new(kernel_num: usize, kernel_radius: usize, kernel_height: usize) -> Self {
        assert!(kernel_height > 0);
        let kernels: Vec<Kernel> = (0..kernel_num)
            .map(|_| Kernel::new_random(kernel_radius, kernel_height))
            .collect();

        Self { kernels }
    }

    pub fn apply(&self, tensor: &Tensor) -> Tensor {
        let mut output_tensor = Tensor::new(tensor.width(), tensor.height(), 0);

        for kernel in &self.kernels {
            let other = Self::convolve(tensor, kernel).unwrap();
            output_tensor.merge(other);
        }

        output_tensor
    }

    fn convolve(tensor: &Tensor, kernel: &Kernel) -> Result<Tensor, String> {
        let t_channels = tensor.channels();
        let channels = kernel.channels();
        if t_channels != channels {
            return Err(format!(
                "Tensor channels != Kernel channels ({}!={})",
                t_channels, channels
            )
            .to_owned());
        }

        let width = tensor.width() as i32;
        let height = tensor.height() as i32;
        let mut new_tensor = Tensor::new(width as usize, height as usize, 1);

        let kd = kernel.diameter() as i32;
        let kr = kd / 2;

        // for each pixel in tensor
        for x in 0..width {
            for y in 0..height {
                let mut sum = 0.0;
                // for each pixel in kernel
                for kx in 0..kd {
                    for ky in 0..kd {
                        // relative to the selected tensor pixel
                        // clamped to stay within bounds of the tensor
                        let rx = (x + kx - kr).clamp(0, width - 1);
                        let ry = (y + ky - kr).clamp(0, height - 1);

                        for ch in 0..channels {
                            println!("rx: {rx}, ry: {ry}, ch: {ch}");
                            let t_val = tensor.get_xyz(rx as usize, ry as usize, ch);
                            let k_val = kernel.get_xyz(rx as usize, ry as usize, ch);

                            sum += t_val * k_val;
                        }
                    }
                }

                new_tensor.set_xyz(x as usize, y as usize, 0, sum);
            }
        }

        Ok(new_tensor)
    }

    // fn convolve(tensor: &Tensor, kernel: &Kernel) -> Result<Tensor, ()> {
    //     let t_channels = tensor.channels();
    //     let k_channels = kernel.channels();

    //     // let rgb_image = image.to_rgb8(); // convert once
    //     // let (width, height) = rgb_image.dimensions();
    //     let width = tensor.width();
    //     let height = tensor.height();
    //     let (w, h) = (width as i32, height as i32);

    //     let k = kernel.diameter() as i32;
    //     let r = k / 2;

    //     let mut out = Tensor::new(width, height, 1);

    //     for y in 0..h {
    //         for x in 0..w {
    //             let mut sum = 0.0;

    //             for ky in 0..k {
    //                 for kx in 0..k {
    //                     // TODO: change to 0 padding
    //                     let ix = (x + kx - r).clamp(0, w - 1);
    //                     let iy = (y + ky - r).clamp(0, h - 1);

    //                     // let pixel = rgb_image.get_pixel(ix as u32, iy as u32);
    //                     let val = tensor.get_xyz(ix.try_into().unwrap(), iy.try_into().unwrap(), 0);
    //                     let kernel_val = kernel.get_xy(kx as usize, ky as usize);

    //                     sum += pixel[0] as f32 * kernel_val[0]
    //                         + pixel[1] as f32 * kernel_val[1]
    //                         + pixel[2] as f32 * kernel_val[2];
    //                 }
    //             }

    //             out.put_xy(x as usize, y as usize, sum);
    //         }
    //     }

    //     out
    // }
}
