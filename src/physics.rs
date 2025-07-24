//! src/physics.rs
//! # 物理法則に基づく損失関数 (理論モデル)
//!
//! このモジュールはPINNの「物理法則情報付き」の部分、すなわち「理論モデル」です。
//! `model.rs`で定義された「学習モデル」が最小化すべき目的関数（損失関数）を提供します。
//!
//! 損失は、モデルの予測が物理的にどれだけ妥当かを評価する複数の要素から構成されます。

use crate::constants::{model_dims, physics::*};
use burn::prelude::*;
use burn::nn::loss::{MseLoss, Reduction};
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

/// ハイブリッド損失関数を計算します。
///
/// この関数は、PINNアプローチの根幹をなすもので、2種類の損失を動的に混合します。
/// 1.  **データ損失**: モデルの予測寸法とFEMシミュレーションによる正解寸法との平均二乗誤差（MSE）。
/// 2.  **物理損失**: モデルの予測寸法が、物理法則（片持ち梁の理論式）や物理的制約にどれだけ従っているかを示す損失。
///
/// # Arguments
///
/// * `predicted_dims` - モデルが予測した音叉の寸法を表すテンソル。
/// * `target_freqs` - 目標周波数を表すテンソル。
/// * `target_dims` - FEMデータセットによる正解の寸法を表すテンソル。
/// * `alpha` - データ損失と物理損失の混合比率を決定する係数（0.0〜1.0）。
/// * `theory` - 使用する梁の物理理論モデル（オイラーまたはティモシェンコ）。
///
/// # Returns
///
/// 計算された合計損失を表すスカラーテンソル。
pub fn tuning_fork_loss<B: Backend>(
    predicted_dims: Tensor<B, 2>,
    target_freqs: Tensor<B, 2>,
    target_dims: Tensor<B, 2>,
    alpha: f32,
    theory: BeamTheory,
) -> Tensor<B, 1> {
    let pi = std::f32::consts::PI;
    let epsilon = 1e-8;

    // --- 各次元のテンソルへの参照を取得 ---
    let dim_tensors = predicted_dims.clone().split(1, 1);
    let handle_length = &dim_tensors[model_dims::HANDLE_LENGTH_IDX];
    let handle_diameter = &dim_tensors[model_dims::HANDLE_DIAMETER_IDX];
    let prong_length = &dim_tensors[model_dims::PRONG_LENGTH_IDX];
    let prong_diameter = &dim_tensors[model_dims::PRONG_DIAMETER_IDX];
    let prong_gap = &dim_tensors[model_dims::PRONG_GAP_IDX];

    // --- 1. 選択された理論に基づき周波数を計算 ---
    let prong_d2 = prong_diameter.clone().powf_scalar(2.0);
    let area = prong_d2.clone() * (pi / 4.0);
    let moment_of_inertia = prong_d2.powf_scalar(2.0) * (pi / 64.0);

    // オイラー・ベルヌーイ理論による周波数 (f_e)
    let stiffness_e = moment_of_inertia.clone() * YOUNGS_MODULUS;
    let density_mass = area.clone() * DENSITY;
    let sqrt_term_e = (stiffness_e / density_mass.add_scalar(epsilon)).sqrt();
    let length_term_e = prong_length.clone().powf_scalar(2.0);
    let euler_freq = sqrt_term_e.mul_scalar(K_FACTOR / (2.0 * pi)) / length_term_e;

    let predicted_freqs = match theory {
        BeamTheory::Euler => euler_freq,
        BeamTheory::Timoshenko => {
            // ティモシェンコ理論による補正
            let shear_modulus = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO));
            let shear_coeff = (6.0 * (1.0 + POISSON_RATIO)) / (7.0 + 6.0 * POISSON_RATIO);
            let correction_factor = (YOUNGS_MODULUS * moment_of_inertia)
                / (shear_coeff * shear_modulus * area * prong_length.clone().powf_scalar(2.0));

            euler_freq / (1.0f32 + 3.0f32 * correction_factor).sqrt()
        }
    };

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

    // --- 3. 物理損失の計算 ---
    let physics_loss = (frequency_loss
        + ratio_penalty * PENALTY_WEIGHT_RATIO
        + (prong_diameter_penalty + prong_length_penalty) * PENALTY_WEIGHT_RANGE
        + (handle_length_penalty + handle_diameter_penalty + prong_gap_penalty)
            * PENALTY_WEIGHT_OTHER)
        .mean();

    // --- 4. データ損失（MSE）の計算 ---
    let data_loss = MseLoss::new().forward(predicted_dims, target_dims, Reduction::Mean);

    // --- 5. 2つの損失をalphaで混合 ---
    let total_loss = data_loss.mul_scalar(alpha) + physics_loss.mul_scalar(1.0 - alpha);

    total_loss
}
