//! src/physics.rs
//! # 物理法則に基づく損失関数 (理論モデル)
//!
//! このモジュールはPINNの「物理法則情報付き」の部分、すなわち「理論モデル」です。
//! `model.rs`で定義された「学習モデル」が最小化すべき目的関数（損失関数）を提供します。
//!
//! 損失は、モデルの予測が物理的にどれだけ妥当かを評価する複数の要素から構成されます。

use crate::constants::{model_dims, physics::*};
use burn::prelude::*;
use burn::tensor::{Tensor, activation::relu};

/// Calculates the normalized penalty for a value that should be within a certain range.
fn relative_range_penalty<B: Backend>(
    value: &Tensor<B, 2>,
    lower_bound: f32,
    upper_bound: f32,
) -> Tensor<B, 2> {
    let lower_penalty = (relu(lower_bound - value.clone()) / lower_bound).powf_scalar(2.0);
    let upper_penalty = (relu(value.clone() - upper_bound) / upper_bound).powf_scalar(2.0);
    lower_penalty + upper_penalty
}

pub fn tuning_fork_loss<B: Backend>(
    predicted_dims: Tensor<B, 2>,
    target_freqs: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let pi = std::f32::consts::PI;
    let epsilon = 1e-8;

    // --- 各次元のテンソルへの参照を取得 ---
    let dim_tensors = predicted_dims.split(1, 1);
    let handle_length = &dim_tensors[model_dims::HANDLE_LENGTH_IDX];
    let handle_diameter = &dim_tensors[model_dims::HANDLE_DIAMETER_IDX];
    let prong_length = &dim_tensors[model_dims::PRONG_LENGTH_IDX];
    let prong_diameter = &dim_tensors[model_dims::PRONG_DIAMETER_IDX];
    let prong_gap = &dim_tensors[model_dims::PRONG_GAP_IDX];

    // --- 1. 周波数損失の計算 ---
    let prong_d2 = prong_diameter.clone().powf_scalar(2.0);
    let area = prong_d2.clone() * (pi / 4.0);
    let moment_of_inertia = prong_d2.powf_scalar(2.0) * (pi / 64.0);
    let stiffness = moment_of_inertia * YOUNGS_MODULUS;
    let density_mass = area * DENSITY;
    let sqrt_term = (stiffness / density_mass.add_scalar(epsilon)).sqrt();
    let length_term = prong_length.clone().powf_scalar(2.0);
    let predicted_freqs = sqrt_term.mul_scalar(K_FACTOR / (2.0 * pi)) / length_term;

    let frequency_loss = ((predicted_freqs - target_freqs.clone())
        / target_freqs.add_scalar(epsilon))
    .powf_scalar(2.0);

    // --- 2. 物理的制約に対する正規化ペナルティの計算 ---
    let ratio_penalty = (relu(handle_length.clone() - prong_length.clone())
        / handle_length.clone().add_scalar(epsilon))
    .powf_scalar(2.0);

    let prong_diameter_penalty = relative_range_penalty(prong_diameter, 0.002, 0.02);
    let prong_length_penalty = relative_range_penalty(prong_length, 0.01, 0.2);
    let handle_length_penalty = relative_range_penalty(handle_length, 0.03, 0.15);
    let handle_diameter_penalty = relative_range_penalty(handle_diameter, 0.005, 0.02);
    let prong_gap_penalty = relative_range_penalty(prong_gap, 0.002, 0.02);

    // --- 3. 合計損失の計算 ---
    let total_loss = (frequency_loss
        + ratio_penalty * PENALTY_WEIGHT_RATIO
        + (prong_diameter_penalty + prong_length_penalty) * PENALTY_WEIGHT_RANGE
        + (handle_length_penalty + handle_diameter_penalty + prong_gap_penalty)
            * PENALTY_WEIGHT_OTHER)
        .mean();

    total_loss
}
