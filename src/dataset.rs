//! src/dataset.rs
use crate::constants::model_dims;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct FemDataItem {
    pub frequency: f32,
    pub prong_length: f32,
    pub prong_diameter: f32,
    pub prong_gap: f32,
    pub handle_length: f32,
    pub handle_diameter: f32,
}

#[derive(Clone, Debug)]
pub struct FemDataset {
    pub items: Vec<FemDataItem>,
}

impl FemDataset {
    pub fn new(path: &str) -> Self {
        let file = File::open(path).expect("Failed to open FEM data file.");
        let reader = BufReader::new(file);
        let items = serde_json::from_reader(reader).expect("Failed to parse FEM data.");
        Self { items }
    }
}

impl Dataset<FemDataItem> for FemDataset {
    fn get(&self, index: usize) -> Option<FemDataItem> {
        self.items.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

pub struct PinnBatcher<B: Backend> {
    _device: B::Device,
}

impl<B: Backend> PinnBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { _device: device }
    }
}

impl<B: Backend> Batcher<B, FemDataItem, (Tensor<B, 2>, Tensor<B, 2>)> for PinnBatcher<B> {
    fn batch(&self, items: Vec<FemDataItem>, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch_size = items.len();

        let mut freqs_flat = Vec::with_capacity(batch_size);
        let mut dims_flat = Vec::with_capacity(batch_size * model_dims::NUM_DIMS);

        for item in items {
            freqs_flat.push(item.frequency);
            dims_flat.extend([
                item.prong_length,
                item.prong_diameter,
                item.prong_gap,
                item.handle_length,
                item.handle_diameter,
            ]);
        }

        // ⭐️⭐️⭐️ 修正点 1: `.as_slice()` を使って&[f32]を渡す ⭐️⭐️⭐️
        let freqs_tensor = Tensor::<B, 1>::from_floats(freqs_flat.as_slice(), device).reshape([-1, 1]);
        
        // ⭐️⭐️⭐️ 修正点 2: NUM_DIMSをi32にキャストする ⭐️⭐️⭐️
        let dims_tensor = Tensor::<B, 1>::from_floats(dims_flat.as_slice(), device).reshape([-1, model_dims::NUM_DIMS as i32]);

        (freqs_tensor, dims_tensor)
    }
}