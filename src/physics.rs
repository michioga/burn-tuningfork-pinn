//! src/physics.rs

use crate::{
    constants::physics::BeamTheory,
    constants::{
        DENSITY, K_FACTOR, PENALTY_WEIGHT_OTHER, PENALTY_WEIGHT_RANGE, PENALTY_WEIGHT_RATIO,
        POISSON_RATIO, YOUNGS_MODULUS, model_dims,
    },
};
use burn::{
    nn::loss::{MseLoss, Reduction},
    prelude::*,
    tensor::activation::relu,
};
use std::f32::consts::PI;

// 損失関数内のヘルパー：相対的な範囲ペナルティ
fn relative_range_penalty<B: Backend>(
    value: &Tensor<B, 2>,
    lower_bound: f32,
    upper_bound: f32,
) -> Tensor<B, 2> {
    let lower_penalty = (relu(lower_bound - value.clone()) / lower_bound).powf_scalar(2.0);
    let upper_penalty = (relu(value.clone() - upper_bound) / upper_bound).powf_scalar(2.0);
    lower_penalty + upper_penalty
}

// 損失関数
pub fn tuning_fork_loss<B: Backend>(
    predicted_dims: Tensor<B, 2>,
    target_freqs: Tensor<B, 2>,
    target_dims: Tensor<B, 2>,
    alpha: f32,
    theory: BeamTheory,
) -> Tensor<B, 1> {
    let epsilon: f32 = 1e-8;

    let data_loss = MseLoss::new()
        .forward(predicted_dims.clone(), target_dims.clone(), Reduction::Mean)
        .reshape([1]);

    let batch_size = predicted_dims.dims()[0];
    let handle_length = predicted_dims.clone().slice([
        0..batch_size,
        model_dims::HANDLE_LENGTH_IDX..model_dims::HANDLE_LENGTH_IDX + 1,
    ]);
    let prong_length = predicted_dims.clone().slice([
        0..batch_size,
        model_dims::PRONG_LENGTH_IDX..model_dims::PRONG_LENGTH_IDX + 1,
    ]);

    // 中間計算でしか使わない変数はここで定義
    let prong_diameter = predicted_dims.clone().slice([
        0..batch_size,
        model_dims::PRONG_DIAMETER_IDX..model_dims::PRONG_DIAMETER_IDX + 1,
    ]);
    let prong_gap = predicted_dims.clone().slice([
        0..batch_size,
        model_dims::PRONG_GAP_IDX..model_dims::PRONG_GAP_IDX + 1,
    ]);
    let handle_diameter = predicted_dims.clone().slice([
        0..batch_size,
        model_dims::HANDLE_DIAMETER_IDX..model_dims::HANDLE_DIAMETER_IDX + 1,
    ]);

    let area = prong_diameter.clone().powf_scalar(2.0).mul_scalar(PI / 4.0);
    let moment_of_inertia = prong_diameter
        .clone()
        .powf_scalar(4.0)
        .mul_scalar(PI / 64.0);

    let stiffness_e = moment_of_inertia.clone().mul_scalar(YOUNGS_MODULUS);
    let density_mass = area.clone().mul_scalar(DENSITY);
    let sqrt_term_e = (stiffness_e / density_mass).sqrt();
    let length_term_e = prong_length.clone().powf_scalar(2.0);
    let euler_freq = sqrt_term_e.mul_scalar(K_FACTOR / (2.0 * PI)) / length_term_e;

    let predicted_freq = match theory {
        BeamTheory::Euler => euler_freq,
        BeamTheory::Timoshenko => {
            let shear_modulus = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO));
            let shear_coeff = (6.0 * (1.0 + POISSON_RATIO)) / (7.0 + 6.0 * POISSON_RATIO);

            let numerator = moment_of_inertia.mul_scalar(YOUNGS_MODULUS);
            let denominator = area
                .mul(prong_length.clone().powf_scalar(2.0))
                .mul_scalar(shear_coeff * shear_modulus);
            let correction_factor = numerator / denominator;

            let timo_freq =
                euler_freq.clone() / (correction_factor.mul_scalar(3.0).add_scalar(1.0)).sqrt();
            timo_freq
        }
    };

    let frequency_loss = ((predicted_freq - target_freqs.clone())
        / target_freqs.add_scalar(epsilon))
    .powf_scalar(2.0);

    let handle_length_penalty = relative_range_penalty(&handle_length, 0.03, 0.15);
    let handle_diameter_penalty = relative_range_penalty(&handle_diameter, 0.005, 0.02);
    let prong_length_penalty = relative_range_penalty(&prong_length, 0.01, 0.2);
    let prong_diameter_penalty = relative_range_penalty(&prong_diameter, 0.002, 0.02);
    let prong_gap_penalty = relative_range_penalty(&prong_gap, 0.002, 0.03);

    // 修正点：引き算で所有権が移動する handle_length をクローンする
    let ratio_penalty = (relu(prong_length - handle_length.clone())
        / handle_length.add_scalar(epsilon))
    .powf_scalar(2.0);

    let physics_loss = (frequency_loss
        + ratio_penalty.mul_scalar(PENALTY_WEIGHT_RATIO)
        + (prong_diameter_penalty + prong_length_penalty).mul_scalar(PENALTY_WEIGHT_RANGE)
        + (handle_length_penalty + handle_diameter_penalty + prong_gap_penalty)
            .mul_scalar(PENALTY_WEIGHT_OTHER))
    .mean()
    .reshape([1]);

    let total_loss = data_loss.mul_scalar(alpha) + physics_loss.mul_scalar(1.0 - alpha);
    total_loss
}
