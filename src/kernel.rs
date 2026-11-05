use thiserror::Error;

use crate::tensor::Tensor;

pub struct Kernel(Tensor);

impl Kernel {
    pub fn new_random(radius: usize, channels: usize) -> Self {
        let diameter = 1 + 2 * radius;
        let tensor = Tensor::new_random(diameter, diameter, channels);

        Self(tensor)
    }

    pub fn diameter(&self) -> usize {
        let width = self.0.width();
        let height = self.0.height();
        assert_eq!(width, height);

        width
    }

    #[inline]
    pub fn channels(&self) -> usize {
        self.0.channels()
    }

    #[inline]
    pub fn get_xyz(&self, x: usize, y: usize, channel: usize) -> f32 {
        self.0.get_xyz(x, y, channel)
    }

    #[inline]
    pub fn inner(&self) -> &Tensor {
        &self.0
    }
}

#[derive(Debug, Error)]
pub enum TensorTryFromKernelError {
    #[error("Expected equal width and height")]
    TensorDimensionsNotEqual,
    #[error("Expeted odd-sized diameter")]
    TensorDiameterNotOdd,
}

impl TryFrom<Tensor> for Kernel {
    type Error = TensorTryFromKernelError;

    fn try_from(value: Tensor) -> Result<Self, Self::Error> {
        let height = value.height();
        let width = value.width();
        if height != width {
            return Err(TensorTryFromKernelError::TensorDimensionsNotEqual);
        } else if height % 2 == 0 {
            return Err(TensorTryFromKernelError::TensorDiameterNotOdd);
        } else {
            return Ok(Self(value));
        }
    }
}
