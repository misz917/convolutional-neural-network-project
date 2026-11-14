use crate::{cnn::CNN, data_loader::DataLoader};

pub struct Trainer {
    data_loader: DataLoader,
    epochs: u32,
    image_size: (usize, usize),
}

impl Trainer {
    pub fn new(
        data_loader: DataLoader,
        model: &mut CNN,
        epochs: u32,
        image_size: (usize, usize),
    ) -> Self {
        Self {
            epochs,
            image_size,
            data_loader,
        }
    }

    pub fn run(&mut self) {
        let option = self.data_loader.next();
        todo!()
    }
}
