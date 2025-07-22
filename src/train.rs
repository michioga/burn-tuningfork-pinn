//! src/train.rs

// ⭐️⭐️⭐️ 修正点: `tuning_fork_loss`と`constants`をインポート ⭐️⭐️⭐️
use crate::{
    model::TuningForkPINN,
    physics::tuning_fork_loss, // 損失関数をインポート
};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataloader::batcher::Batcher, dataset::Dataset},
    lr_scheduler::constant::ConstantLr,
    module::Module,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::{AutodiffBackend, Backend},
    train::{LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep},
};
use rand::{Rng, thread_rng};

// ...（TuningForkDatasetとTuningForkBatcherは変更なし）...
#[derive(Clone, Debug)]
pub struct TuningForkDataset {
    pub size: usize,
    pub freq_range: (f32, f32),
}

impl Dataset<f32> for TuningForkDataset {
    fn get(&self, _index: usize) -> Option<f32> {
        let mut rng = thread_rng();
        let frequency = rng.gen_range(self.freq_range.0..=self.freq_range.1);
        Some(frequency)
    }

    fn len(&self) -> usize {
        self.size
    }
}

pub struct TuningForkBatcher<B: Backend> {
    _device: B::Device,
}

impl<B: Backend> TuningForkBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { _device: device }
    }
}

impl<B: Backend> Batcher<B, f32, Tensor<B, 2>> for TuningForkBatcher<B> {
    fn batch(&self, items: Vec<f32>, device: &B::Device) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(items.as_slice(), device).reshape([-1, 1])
    }
}


impl<B: AutodiffBackend> TrainStep<Tensor<B, 2>, RegressionOutput<B>> for TuningForkPINN<B> {
    fn step(&self, item: Tensor<B, 2>) -> TrainOutput<RegressionOutput<B>> {
        let predicted_dims = self.forward(item.clone());
        let loss = tuning_fork_loss(predicted_dims.clone(), item.clone());
        let output = RegressionOutput {
            loss: loss.clone(),
            output: predicted_dims,
            targets: item,
        };
        TrainOutput::new(self, loss.backward(), output)
    }
}

impl<B: Backend> ValidStep<Tensor<B, 2>, RegressionOutput<B>> for TuningForkPINN<B> {
    fn step(&self, item: Tensor<B, 2>) -> RegressionOutput<B> {
        let predicted_dims = self.forward(item.clone());
        // ⭐️⭐️⭐️ 修正点: ValidStep内でも損失関数を呼び出す ⭐️⭐️⭐️
        let loss = tuning_fork_loss(predicted_dims.clone(), item.clone());
        RegressionOutput {
            loss,
            output: predicted_dims,
            targets: item,
        }
    }
}

// ...（TrainingConfigとrun関数は変更なし）...
#[derive(Config)]
pub struct TrainingConfig {
    pub optimizer: AdamConfig,
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    #[config(default = 2000)]
    pub num_epochs: usize,
    #[config(default = 1024)]
    pub batch_size: usize,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let config = TrainingConfig::new(AdamConfig::new());
    let artifact_dir = "./artifacts";
    std::fs::create_dir_all(artifact_dir).ok();

    let batcher_train = TuningForkBatcher::<B>::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(8) // CPUコア数に応じて調整
        .build(TuningForkDataset {
            size: config.batch_size * 100,
            freq_range: (200.0, 1800.0),
        });

    let batcher_valid = TuningForkBatcher::<B::InnerBackend>::new(device.clone());
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .num_workers(8) // CPUコア数に応じて調整
        .build(TuningForkDataset {
            size: config.batch_size * 20,
            freq_range: (1800.0, 2000.0),
        });

    let scheduler = ConstantLr::new(config.learning_rate);

    let learner = LearnerBuilder::new(artifact_dir)
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .grads_accumulation(1)
        .build(
            TuningForkPINN::<B>::new(&device),
            config.optimizer.init(),
            scheduler,
        );

    println!("🚀 Starting training on {:?}...", device);
    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    let model_record = model_trained.into_record();
    CompactRecorder::new()
        .record(model_record, format!("{artifact_dir}/model").into())
        .expect("Failed to save trained model");

    println!("\n✅ Model saved to '{artifact_dir}/model.mpk'");
}