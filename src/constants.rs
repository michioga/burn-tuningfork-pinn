//! src/constants.rs

// 修正点：不要な Module を削除
// use burn::prelude::Module;

pub mod model_dims {
    pub const NUM_DIMS: usize = 5;
    pub const HANDLE_LENGTH_IDX: usize = 0;
    pub const HANDLE_DIAMETER_IDX: usize = 1;
    pub const PRONG_LENGTH_IDX: usize = 2;
    pub const PRONG_DIAMETER_IDX: usize = 3;
    pub const PRONG_GAP_IDX: usize = 4;
}

pub mod physics {
    #[derive(Debug, Clone, Copy)]
    pub enum BeamTheory {
        Euler,
        Timoshenko,
    }

    // 使用する梁理論を選択
    pub const BEAM_THEORY_CHOICE: BeamTheory = BeamTheory::Timoshenko;
}

// --- 物理定数 ---
/// ヤング率 (Pa) - 鋼
pub const YOUNGS_MODULUS: f32 = 206e9;
/// 密度 (kg/m^3) - 鋼
pub const DENSITY: f32 = 7850.0;
/// ポアソン比 - 鋼
pub const POISSON_RATIO: f32 = 0.3;
/// 1次曲げ振動モードの定数
pub const K_FACTOR: f32 = 3.5160;

// --- 損失関数の重み設定 ---
/// `ratio_penalty`（プロング長 > 柄長）に対する重み。
pub const PENALTY_WEIGHT_RATIO: f32 = 30.0;
/// `range_penalty`（プロング関連の寸法範囲）に対する重み。
pub const PENALTY_WEIGHT_RANGE: f32 = 30.0;
/// `range_penalty`（その他の寸法範囲）に対する重み。
pub const PENALTY_WEIGHT_OTHER: f32 = 50.0;
