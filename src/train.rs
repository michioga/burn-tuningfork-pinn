//! src/train.rs

use crate::{dataset::{FemDataset, PinnBatcher}, model::TuningForkPINN};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    lr_scheduler::constant::ConstantLr,
    module::Module,
    nn::loss::{MseLoss, Reduction},
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::{AutodiffBackend, Backend},
    train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

impl<B: AutodiffBackend> TrainStep<(Tensor<B, 2>, Tensor<B, 2>), RegressionOutput<B>>
    for TuningForkPINN<B>
{
    fn step(&self, item: (Tensor<B, 2>, Tensor<B, 2>)) -> TrainOutput<RegressionOutput<B>> {
        let (freqs, targets) = item;
        let predicted_dims = self.forward(freqs);
        let loss = MseLoss::new().forward(predicted_dims.clone(), targets.clone(), Reduction::Mean);

        let output = RegressionOutput {
            loss: loss.clone(),
            output: predicted_dims,
            targets,
        };
        TrainOutput::new(self, loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<(Tensor<B, 2>, Tensor<B, 2>), RegressionOutput<B>> for TuningForkPINN<B> {
    fn step(&self, item: (Tensor<B, 2>, Tensor<B, 2>)) -> RegressionOutput<B> {
        let (freqs, targets) = item;
        let predicted_dims = self.forward(freqs);
        let loss = MseLoss::new().forward(predicted_dims.clone(), targets.clone(), Reduction::Mean);
        RegressionOutput {
            loss,
            output: predicted_dims,
            targets,
        }
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamConfig,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    #[config(default = 300)]
    pub num_epochs: usize,
    #[config(default = 3)]
    pub batch_size: usize,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let config = TrainingConfig::new(AdamConfig::new());
    let artifact_dir = "./artifacts";
    std::fs::create_dir_all(artifact_dir).ok();

    let dataset_all = FemDataset::new("data/fem_data_augmented.csv");

    let dataset_train = dataset_all.clone();
    let dataset_valid = dataset_all;

    let batcher_train = PinnBatcher::<B>::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(dataset_train);

    let batcher_valid = PinnBatcher::<B::InnerBackend>::new(device.clone());
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(4)
        .build(dataset_valid);

    let scheduler = ConstantLr::new(config.learning_rate);

    let learner = LearnerBuilder::new(artifact_dir)
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            TuningForkPINN::<B>::new(&device),
            config.optimizer.init(),
            scheduler,
        );

    println!("🚀 Starting training on {:?}...", device);
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    let model_record = model_trained.into_record();
    CompactRecorder::new()
        .record(model_record, format!("{}/model", artifact_dir).into())
        .expect("Failed to save trained model");

    println!("\n✅ Model saved to '{}/model.mpk'", artifact_dir);
}
