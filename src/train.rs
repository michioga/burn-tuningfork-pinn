//! # モデルの学習モジュール
//!
//! `LearnerBuilder`を使用して学習プロセスを構成し、
//! データセットを使ってモデルの学習を実行します。

// ⭐️ 1. physicsモジュールをインポート
use crate::{
    constants::physics::BEAM_THEORY_CHOICE,
    dataset::{FemDataset, PinnBatcher},
    model::TuningForkPINN,
    physics,
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

/// モデルの学習ステップを定義します。
///
/// この実装はPINNアプローチの心臓部です。
/// `physics::tuning_fork_loss` を呼び出し、データ損失（FEMデータとの誤差）と
/// 物理損失（物理法則との誤差）を組み合わせたハイブリッド損失を計算します。
/// この損失を最小化することで、モデルはデータへの忠実度と物理法則への準拠を両立するように学習します。
impl<B: AutodiffBackend> TrainStep<(Tensor<B, 2>, Tensor<B, 2>), RegressionOutput<B>>
    for TuningForkPINN<B>
{
    /// モデルの検証ステップの実装
    fn step(&self, item: (Tensor<B, 2>, Tensor<B, 2>)) -> TrainOutput<RegressionOutput<B>> {
        let (freqs, targets) = item; // `targets` がFEMの正解寸法
        let predicted_dims = self.forward(freqs.clone());

        // FEMデータをより重視したい場合、alphaを0.5などに設定
        const ALPHA: f32 = 0.5;

        let loss = physics::tuning_fork_loss(
            predicted_dims.clone(),
            freqs,
            targets.clone(), // 正解寸法を渡す
            ALPHA,           // alphaを渡す
            BEAM_THEORY_CHOICE, // 梁理論の選択を渡す
        );

        let output = RegressionOutput {
            loss: loss.clone(),
            output: predicted_dims,
            targets,
        };
        TrainOutput::new(self, loss.backward(), output)
    }
}

/// モデルの検証ステップを定義します。
///
/// 学習ステップと同様に、ハイブリッド損失（データ損失 + 物理損失）を計算して、
/// 検証データに対するモデルの性能を評価します。
impl<B: Backend> ValidStep<(Tensor<B, 2>, Tensor<B, 2>), RegressionOutput<B>>
    for TuningForkPINN<B>
{
    ///
    fn step(&self, item: (Tensor<B, 2>, Tensor<B, 2>)) -> RegressionOutput<B> {
        let (freqs, targets) = item;
        let predicted_dims = self.forward(freqs.clone());

        const ALPHA: f32 = 0.5;

        // こちらも物理損失関数で評価する
        let loss = physics::tuning_fork_loss(
            predicted_dims.clone(),
            freqs,
            targets.clone(),
            ALPHA,
            BEAM_THEORY_CHOICE,
        );

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
