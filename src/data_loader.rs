use std::{
    collections::{BTreeMap, HashMap},
    fs, io,
    path::PathBuf,
};

use image::DynamicImage;
use tracing::{error, info, trace, warn};

pub struct DataLoader {
    targets: HashMap<String, Vec<f32>>,
    data_dir: PathBuf,
    classes: BTreeMap<String, Vec<PathBuf>>,
    class_i: usize,
    image_i: usize,
}

impl DataLoader {
    pub fn new(data_dir: PathBuf) -> Self {
        let mut this = Self {
            data_dir,
            classes: BTreeMap::new(),
            class_i: 0,
            image_i: 0,
            targets: HashMap::new(),
        };
        Self::populate_classes(&mut this).unwrap();
        Self::prepare_targets(&mut this);
        info!(
            "Populated DataLoader.classes with {} directories and {} image paths",
            this.num_classes(),
            this.num_images()
        );
        if this.num_classes() == 0 || this.num_images() == 0 {
            error!("No data was loaded");
        }

        this
    }

    fn prepare_targets(&mut self) {
        let zero_vec = vec![0.0f32; self.num_classes()];

        for (i, class) in self.get_all_classes().iter().enumerate() {
            let mut target = zero_vec.clone();
            target[i] = 1.0;

            self.targets.insert(class.to_owned(), target);
        }
    }

    fn populate_classes(&mut self) -> io::Result<()> {
        for directory in fs::read_dir(&self.data_dir)? {
            let directory = directory?;
            let directory_str: String = directory.file_name().to_string_lossy().into();
            trace!("Loading directory {:?}", directory);

            let mut limit = 0;

            if directory.file_type()?.is_dir() {
                for image_file in fs::read_dir(directory.path())? {
                    let image_file = image_file?;
                    trace!("Loading image {:?}", image_file);

                    limit += 1;
                    if limit % 10 != 0 {
                        continue;
                    }

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

    pub fn get_targets(&self) -> &HashMap<String, Vec<f32>> {
        &self.targets
    }

    pub fn num_classes(&self) -> usize {
        self.classes.len()
    }

    pub fn num_images(&self) -> usize {
        self.classes.iter().map(|c| c.1.len()).sum()
    }

    pub fn get_all_classes(&self) -> Vec<String> {
        self.classes.keys().cloned().collect()
    }

    pub fn reset(&mut self) {
        self.class_i = 0;
        self.image_i = 0;
    }
}

impl Iterator for DataLoader {
    type Item = (DynamicImage, String);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (class_name, paths) = self.classes.iter().nth(self.class_i)?;

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
