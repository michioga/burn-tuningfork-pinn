//! # モデルの学習モジュール
//!
//! `LearnerBuilder`を使用して学習プロセスを構成し、
//! データセットを使ってモデルの学習を実行します。

// ⭐️ 1. physicsモジュールをインポート
use crate::{
    dataset::{FemDataset, PinnBatcher},
    model::TuningForkPINN,
    physics, // <--- ADD THIS
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    lr_scheduler::constant::ConstantLr,
    module::Module,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        LearnerBuilder, RegressionOutput, TrainOutput, TrainStep, ValidStep,
        metric::{LearningRateMetric, LossMetric},
    },
};

/// モデルの学習ステップの実装
///
/// ここがPINN（物理法則情報付きニューラルネットワーク）アプローチの心臓部です。
/// 通常の教師あり学習のように、モデルの出力をデータセットの「正解」と比較する代わりに、
/// `physics::tuning_fork_loss` を使って、予測が物理法則にどれだけ従っているかを評価します。
/// この「物理損失」を最小化することで、モデルは物理法則そのものを学習します。
impl<B: AutodiffBackend> TrainStep<(Tensor<B, 2>, Tensor<B, 2>), RegressionOutput<B>>
    for TuningForkPINN<B>
{
    /// モデルの検証ステップの実装
    fn step(&self, item: (Tensor<B, 2>, Tensor<B, 2>)) -> TrainOutput<RegressionOutput<B>> {
        let (freqs, targets) = item; // `targets` はログ出力用に残すが、損失計算には使わない
        let predicted_dims = self.forward(freqs.clone());

        // MseLossの代わりに、物理損失関数を呼び出す
        let loss = physics::tuning_fork_loss(predicted_dims.clone(), freqs);

        let output = RegressionOutput {
            loss: loss.clone(),
            output: predicted_dims,
            targets,
        };
        TrainOutput::new(self, loss.backward(), output)
    }
}

/// モデルの検証ステップの実装
impl<B: Backend> ValidStep<(Tensor<B, 2>, Tensor<B, 2>), RegressionOutput<B>>
    for TuningForkPINN<B>
{
    ///
    fn step(&self, item: (Tensor<B, 2>, Tensor<B, 2>)) -> RegressionOutput<B> {
        let (freqs, targets) = item;
        let predicted_dims = self.forward(freqs.clone());

        // こちらも物理損失関数で評価する
        let loss = physics::tuning_fork_loss(predicted_dims.clone(), freqs);

        RegressionOutput {
            loss,
            output: predicted_dims,
            targets,
        }
    }
}

/// 学習設定
#[derive(Config)]
pub struct TrainingConfig {
    /// 使用するオプティマイザの設定
    pub optimizer: AdamConfig,
    /// 学習率
    #[config(default = 1e-4)]
    pub learning_rate: f64,
    /// 学習エポック数
    #[config(default = 300)]
    pub num_epochs: usize,
    /// バッチサイズ
    #[config(default = 32)]
    pub batch_size: usize,
}

// ... run()関数は変更なし ...
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
        .metric_train_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_valid_numeric(LossMetric::new())
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
