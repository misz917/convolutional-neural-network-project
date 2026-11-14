use std::time::Instant;

use image::GenericImageView;
use tracing::{info, warn};

use crate::{cnn::CNN, data_loader::DataLoader, tensor::Tensor};

pub struct Trainer<'a> {
    model: &'a mut CNN,
    data_loader: DataLoader,
    image_size: (u32, u32),
    epochs: u32,
    learning_rate: f32,
}

impl<'a> Trainer<'a> {
    pub fn new(
        model: &'a mut CNN,
        data_loader: DataLoader,
        image_size: (u32, u32),
        epochs: u32,
        learning_rate: f32,
    ) -> Self {
        Self {
            epochs,
            image_size,
            data_loader,
            learning_rate,
            model,
        }
    }

    pub fn run(&mut self) {
        let instant = Instant::now();
        info!("Starting training");

        for epoch in 0..self.epochs {
            let instant = Instant::now();
            info!("Starting epoch {}", epoch);

            while let Some((image, class_name)) = self.data_loader.next() {
                if self.image_size != image.dimensions() {
                    warn!(
                        "Skipped image with size other than specified, class: {}, size: {:?}",
                        class_name,
                        image.dimensions()
                    );
                    continue;
                }

                let target = self.data_loader.get_targets().get(&class_name).unwrap();
                let prediction = self.model.forward(image.clone());
                info!("Prediction: {:?}", prediction);
                let gradient = mse_gradient(&prediction, target);
                info!("Gradient: {:?}", gradient);
                self.model.backward(gradient);
                self.model.update(self.learning_rate);
            }

            self.data_loader.reset();

            let elapsed = instant.elapsed();
            info!(
                "Ending epoch {}, elapsed: {}s",
                epoch,
                elapsed.as_secs_f32()
            );
        }

        let elapsed = instant.elapsed();
        info!("Ending training, elapsed: {}s", elapsed.as_secs_f32());
    }
}

fn mse_gradient(prediction: &[f32], target: &[f32]) -> Tensor {
    assert_eq!(prediction.len(), target.len());
    assert!(prediction.len() > 0);

    let n = prediction.len() as f32;

    let gradient: Box<[f32]> = prediction
        .iter()
        .zip(target)
        .map(|(p, t)| 2.0 * (p - t) / n)
        .collect();

    Tensor::from(gradient)
}
