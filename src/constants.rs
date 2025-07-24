//! # 定数モジュール
//!
//! このモジュールは、物理計算やモデル定義で使用される定数を定義します。

/// 物理計算に関する定数
pub mod physics {
    // 材質: 一般構造用圧延鋼材(SS400) を想定
    /// ヤング率 (Pa)。材料の硬さを示す指標。
    pub const YOUNGS_MODULUS: f32 = 206.0e9;
    /// 密度 (kg/m^3)。
    pub const DENSITY: f32 = 7850.0;
    /// ポアソン比。材料が引張られた際の横方向の縮みを示す。
    pub const POISSON_RATIO: f32 = 0.3;

    /// 片持ち梁の1次曲げ振動モードを計算するための無次元化定数(λ^2)
    pub const K_FACTOR: f32 = 3.5160;

    // --- 損失関数の重み設定 ---
    /// `ratio_penalty`（プロング長 > 柄長）に対する重み。
    pub const PENALTY_WEIGHT_RATIO: f32 = 0.5;
    /// `range_penalty`（プロング関連の寸法範囲）に対する重み。
    pub const PENALTY_WEIGHT_RANGE: f32 = 1.0;
    /// `range_penalty`（その他の寸法範囲）に対する重み。
    pub const PENALTY_WEIGHT_OTHER: f32 = 1.0;

    // --- 物理モデルの選択 ---
    /// 使用する梁の理論モデルを定義します。
    #[derive(Clone, Copy, Debug)]
    pub enum BeamTheory {
        /// オイラー・ベルヌーイの梁理論（せん断変形を無視する単純なモデル）
        Euler,
        /// ティモシェンコの梁理論（せん断変形を考慮する高精度なモデル）
        Timoshenko,
    }

    /// プロジェクト全体で使用する梁理論を選択します。
    pub const BEAM_THEORY_CHOICE: BeamTheory = BeamTheory::Timoshenko;
}

/// モデルの寸法に関する定数
pub mod model_dims {
    /// 出力次元の総数
    pub const NUM_DIMS: usize = 5;
    /// 柄の長さのインデックス
    pub const HANDLE_LENGTH_IDX: usize = 0;
    /// 柄の直径のインデックス
    pub const HANDLE_DIAMETER_IDX: usize = 1;
    /// プロングの長さのインデックス
    pub const PRONG_LENGTH_IDX: usize = 2;
    /// プロングの直径のインデックス
    pub const PRONG_DIAMETER_IDX: usize = 3;
    /// プロングの間隔のインデックス
    pub const PRONG_GAP_IDX: usize = 4;
}

