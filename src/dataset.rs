//! src/dataset.rs
use crate::constants::model_dims;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use serde::{Deserialize, Serialize};

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
        let mut rdr = csv::Reader::from_path(path).expect("Failed to open FEM data file.");
        let items = rdr
            .deserialize()
            .collect::<Result<Vec<FemDataItem>, csv::Error>>()
            .expect("Failed to parse FEM data.");
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

        let freqs_tensor = Tensor::<B, 1>::from_floats(freqs_flat.as_slice(), device).reshape([-1, 1]);
        let dims_tensor = Tensor::<B, 1>::from_floats(dims_flat.as_slice(), device).reshape([-1, model_dims::NUM_DIMS as i32]);

        (freqs_tensor, dims_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{NdArray, NdArrayDevice};

    #[test]
    fn test_fem_dataset_loading() {
        // Create a dummy CSV file for testing
        let csv_data = "frequency,prong_length,prong_diameter,prong_gap,handle_length,handle_diameter\n\
                        440.0,0.1,0.005,0.01,0.05,0.006\n\
                        256.0,0.12,0.006,0.012,0.06,0.007\n";
        let file_path = "test_fem_data.csv";
        std::fs::write(file_path, csv_data).unwrap();

        let dataset = FemDataset::new(file_path);
        assert_eq!(dataset.len(), 2);
        assert_eq!(dataset.get(0).unwrap().frequency, 440.0);

        // Clean up the dummy file
        std::fs::remove_file(file_path).unwrap();
    }

    #[test]
    fn test_pinn_batcher() {
        type TestBackend = NdArray<f32>;
        let device = NdArrayDevice::default();
        let batcher = PinnBatcher::<TestBackend>::new(device.clone());

        let items = vec![
            FemDataItem {
                frequency: 440.0,
                prong_length: 0.1,
                prong_diameter: 0.005,
                prong_gap: 0.01,
                handle_length: 0.05,
                handle_diameter: 0.006,
            },
            FemDataItem {
                frequency: 256.0,
                prong_length: 0.12,
                prong_diameter: 0.006,
                prong_gap: 0.012,
                handle_length: 0.06,
                handle_diameter: 0.007,
            },
        ];

        let (freqs, dims) = batcher.batch(items, &device);

        assert_eq!(freqs.dims(), [2, 1]);
        assert_eq!(dims.dims(), [2, 5]);

        let freqs_data = freqs.to_data();
        let dims_data = dims.to_data();

        assert_eq!(freqs_data.value, vec![440.0, 256.0]);
        assert_eq!(
            dims_data.value,
            vec![0.1, 0.005, 0.01, 0.05, 0.006, 0.12, 0.006, 0.012, 0.06, 0.007]
        );
    }
}
