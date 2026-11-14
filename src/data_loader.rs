use std::{collections::HashMap, path::PathBuf};

use image::DynamicImage;
use tracing::{trace, warn};

pub struct DataLoader {
    data_dir: PathBuf,
    auto_reset: bool,
    dataset_structure: HashMap<String, Vec<PathBuf>>,
    class_num: usize,
    image_num: usize,
}

impl DataLoader {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            data_dir,
            auto_reset: false,
            dataset_structure: HashMap::new(),
            class_num: 0,
            image_num: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.dataset_structure.len()
    }

    pub fn get_all_classes(&self) -> Vec<String> {
        self.dataset_structure.keys().cloned().collect()
    }

    pub fn next(&mut self) -> Option<(DynamicImage, String)> {
        let option = self.dataset_structure.iter().nth(self.class_num);
        if option.is_none() {
            trace!("No next image found");

            if self.auto_reset {
                trace!("Auto reset triggered");
                self.reset();
                return self.next();
            }
        }

        let (class_name, image_paths) = option?;

        if !(self.image_num < image_paths.len()) {
            self.image_num = 0;
            self.class_num += 1;
            return self.next();
        }

        let image_path = &image_paths[self.image_num];
        let image = match image::open(&image_path) {
            Ok(ok) => {
                trace!("Next image loaded: {}", image_path.to_string_lossy());
                ok
            }
            Err(err) => {
                warn!("Error while loading image: {}", err);
                return self.next();
            }
        };

        Some((image, class_name.to_owned()))
    }

    pub fn auto_reset(&mut self, active: bool) {
        self.auto_reset = active;
    }

    pub fn reset(&mut self) {
        self.class_num = 0;
        self.image_num = 0;
    }
}
