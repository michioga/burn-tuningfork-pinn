//! src/train.rs

use crate::{
    constants::physics as physics_consts,
    dataset::{FemDataset, PinnBatcher},
    model::TuningForkPINN,
    physics,
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep, metric::LossMetric,
    },
};

const ARTIFACT_DIR: &str = "./artifacts";

// ALPHAを0.5に設定 (1.0に近いほどCSVのデータを優先する)
const ALPHA: f32 = 1.0;

impl<B: AutodiffBackend> TrainStep<(Tensor<B, 2>, Tensor<B, 2>), RegressionOutput<B>>
    for TuningForkPINN<B>
{
    fn step(&self, item: (Tensor<B, 2>, Tensor<B, 2>)) -> TrainOutput<RegressionOutput<B>> {
        let (freqs, targets) = item;
        let predicted_dims = self.forward(freqs.clone());

        let loss = physics::tuning_fork_loss(
            predicted_dims.clone(),
            freqs,
            targets.clone(),
            ALPHA,
            physics_consts::BEAM_THEORY_CHOICE,
        );
        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
            RegressionOutput::new(loss, predicted_dims, targets),
        )
    }
}

impl<B: Backend> ValidStep<(Tensor<B, 2>, Tensor<B, 2>), RegressionOutput<B>>
    for TuningForkPINN<B>
{
    fn step(&self, item: (Tensor<B, 2>, Tensor<B, 2>)) -> RegressionOutput<B> {
        let (freqs, targets) = item;
        let predicted_dims = self.forward(freqs.clone());

        let loss = physics::tuning_fork_loss(
            predicted_dims.clone(),
            freqs,
            targets.clone(),
            ALPHA,
            physics_consts::BEAM_THEORY_CHOICE,
        );

        RegressionOutput::new(loss, predicted_dims, targets)
    }
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let dataset_all = FemDataset::new("data/summary_parameters.csv");

    let mut all_items = dataset_all.items;
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    all_items.shuffle(&mut rng);

    let split_index = (all_items.len() as f32 * 0.8).round() as usize;
    let (train_items, test_items) = all_items.split_at(split_index);

    let dataset_train = FemDataset {
        items: train_items.to_vec(),
    };
    let dataset_test = FemDataset {
        items: test_items.to_vec(),
    };

    let model = TuningForkPINN::new(&device);

    let batcher_train = PinnBatcher::<B>::new(device.clone());
    let batcher_valid = PinnBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(32)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(32)
        .shuffle(42)
        .num_workers(4)
        .build(dataset_test);

    // 学習の設定
    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(300)
        .build(
            model,
            AdamConfig::new()
                .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
                .init(),
            1e-5, // 学習率を1e-5に設定
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{ARTIFACT_DIR}/model"), &CompactRecorder::new())
        .expect("Failed to save trained model");
}
