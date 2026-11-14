use std::{collections::BTreeMap, fs, io, path::PathBuf};

use image::DynamicImage;
use tracing::{trace, warn};

pub struct DataLoader {
    data_dir: PathBuf,
    auto_reset: bool,
    classes: BTreeMap<String, Vec<PathBuf>>,
    class_i: usize,
    image_i: usize,
}

impl DataLoader {
    pub fn new(data_dir: PathBuf) -> Self {
        let mut this = Self {
            data_dir,
            auto_reset: false,
            classes: BTreeMap::new(),
            class_i: 0,
            image_i: 0,
        };
        Self::populate_classes(&mut this).unwrap();

        this
    }

    fn populate_classes(&mut self) -> io::Result<()> {
        for directory in fs::read_dir(&self.data_dir)? {
            let directory = directory?;
            let directory_str: String = directory.file_name().to_string_lossy().into();

            if directory.file_type()?.is_dir() {
                for image_file in fs::read_dir(directory.path())? {
                    let image_file = image_file?;

                    if image_file.file_type()?.is_file() {
                        let image_file_path = image_file.path();
                        self.classes
                            .entry(directory_str.clone())
                            .and_modify(|v| v.push(image_file_path.clone()))
                            .or_insert_with(|| vec![image_file_path]);
                    }
                }
            }
        }

        Ok(())
    }

    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    pub fn get_all_classes(&self) -> Vec<String> {
        self.classes.keys().cloned().collect()
    }

    pub fn auto_reset(&mut self, active: bool) {
        self.auto_reset = active;
    }

    pub fn reset(&mut self) {
        self.class_i = 0;
        self.image_i = 0;
    }
}

impl Iterator for DataLoader {
    type Item = (DynamicImage, String);

    fn next(&mut self) -> Option<Self::Item> {
        let mut wrapped = false;

        loop {
            let Some((class_name, paths)) = self.classes.iter().nth(self.class_i) else {
                if !self.auto_reset || wrapped {
                    trace!("No next image found");
                    return None;
                }
                trace!("Auto reset triggered");
                self.reset();
                wrapped = true;
                continue;
            };

            if self.image_i >= paths.len() {
                self.image_i = 0;
                self.class_i += 1;
                continue;
            }

            let path = &paths[self.image_i];
            self.image_i += 1;

            match image::open(path) {
                Ok(img) => {
                    trace!("Next image loaded: {}", path.display());
                    return Some((img, class_name.clone()));
                }
                Err(err) => warn!("Error loading {}: {err}", path.display()),
            }
        }
    }
}
