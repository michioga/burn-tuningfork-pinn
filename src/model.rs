//! src/model.rs
use crate::constants::model_dims;
use burn::{
    prelude::*,
    nn::{Linear, LinearConfig, Relu, LayerNorm, LayerNormConfig},
    tensor::activation::softplus,
};

#[derive(Module, Debug)]
pub struct TuningForkPINN<B: Backend> {
    layer_1: Linear<B>,
    norm_1: LayerNorm<B>,
    activation_1: Relu,
    layer_2: Linear<B>,
    norm_2: LayerNorm<B>,
    activation_2: Relu,
    output_layer: Linear<B>,
}

impl<B: Backend> TuningForkPINN<B> {
    pub fn new(device: &B::Device) -> Self {
        let hidden_size = 256;
        Self {
            layer_1: LinearConfig::new(1, hidden_size).init(device),
            norm_1: LayerNormConfig::new(hidden_size).init(device),
            activation_1: Relu::new(),
            layer_2: LinearConfig::new(hidden_size, hidden_size).init(device),
            norm_2: LayerNormConfig::new(hidden_size).init(device),
            activation_2: Relu::new(),
            output_layer: LinearConfig::new(hidden_size, model_dims::NUM_DIMS).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer_1.forward(input);
        let x = self.norm_1.forward(x);
        let x = self.activation_1.forward(x);
        let x = self.layer_2.forward(x);
        let x = self.norm_2.forward(x);
        let x = self.activation_2.forward(x);
        let x = self.output_layer.forward(x);

        softplus(x, 1.0)
    }
}