//! src/model.rs
use burn::{
    nn::{
        // Dropout関連のモジュールをインポート
        Dropout,
        DropoutConfig,
        LayerNorm,
        LayerNormConfig,
        Linear,
        LinearConfig,
        Relu,
    },
    prelude::*,
    tensor::activation::softplus,
};

use crate::constants::model_dims;

#[derive(Module, Debug)]
pub struct TuningForkPINN<B: Backend> {
    // 層を増やし、Dropout層を追加
    layer_1: Linear<B>,
    norm_1: LayerNorm<B>,
    activation_1: Relu,
    dropout_1: Dropout,

    layer_2: Linear<B>,
    norm_2: LayerNorm<B>,
    activation_2: Relu,
    dropout_2: Dropout,

    layer_3: Linear<B>,
    norm_3: LayerNorm<B>,
    activation_3: Relu,
    dropout_3: Dropout,

    layer_4: Linear<B>,
    norm_4: LayerNorm<B>,
    activation_4: Relu,
    dropout_4: Dropout,

    output_layer: Linear<B>,
}

impl<B: Backend> TuningForkPINN<B> {
    pub fn new(device: &B::Device) -> Self {
        let hidden_size = 256;
        let dropout_prob = 0.2; // 20%のニューロンを無効化

        Self {
            layer_1: LinearConfig::new(1, hidden_size).init(device),
            norm_1: LayerNormConfig::new(hidden_size).init(device),
            activation_1: Relu::new(),
            dropout_1: DropoutConfig::new(dropout_prob).init(),

            layer_2: LinearConfig::new(hidden_size, hidden_size).init(device),
            norm_2: LayerNormConfig::new(hidden_size).init(device),
            activation_2: Relu::new(),
            dropout_2: DropoutConfig::new(dropout_prob).init(),

            layer_3: LinearConfig::new(hidden_size, hidden_size).init(device),
            norm_3: LayerNormConfig::new(hidden_size).init(device),
            activation_3: Relu::new(),
            dropout_3: DropoutConfig::new(dropout_prob).init(),

            layer_4: LinearConfig::new(hidden_size, hidden_size).init(device),
            norm_4: LayerNormConfig::new(hidden_size).init(device),
            activation_4: Relu::new(),
            dropout_4: DropoutConfig::new(dropout_prob).init(),

            output_layer: LinearConfig::new(hidden_size, model_dims::NUM_DIMS).init(device),
        }
    }

    // forwardパスを新しい層構成に合わせて更新
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        x = self.layer_1.forward(x);
        x = self.norm_1.forward(x);
        x = self.activation_1.forward(x);
        x = self.dropout_1.forward(x);

        x = self.layer_2.forward(x);
        x = self.norm_2.forward(x);
        x = self.activation_2.forward(x);
        x = self.dropout_2.forward(x);

        x = self.layer_3.forward(x);
        x = self.norm_3.forward(x);
        x = self.activation_3.forward(x);
        x = self.dropout_3.forward(x);

        x = self.layer_4.forward(x);
        x = self.norm_4.forward(x);
        x = self.activation_4.forward(x);
        x = self.dropout_4.forward(x);

        let x = self.output_layer.forward(x);

        // ハード制約は使わず、ペナルティで制約をかけるためsoftplusに戻す
        softplus(x, 1.0)
    }
}
