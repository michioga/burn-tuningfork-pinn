# 音叉 設計PINN (Tuning Fork Design PINN)

このプロジェクトは、[Burn](https://burn.dev/)フレームワークを使用して構築された、**物理法則情報付きニューラルネットワーク（PINN）** の実装です。
目標とする周波数（Hz）を入力すると、その周波数を実現するための音叉の最適な寸法を予測する**逆設計問題**を解決します。

## 特徴

  - **物理法則に基づく学習**: モデルの損失関数に物理法則（片持ち梁の振動方程式）を組み込むことで、少ないデータでも物理的に妥当な結果を生成します。
  - **逆問題解決**: 通常のシミュレーション（寸法→周波数）とは逆に、目標の周波数から最適な寸法を予測します。
  - **マルチバックエンド対応**: コマンドライン引数で計算バックエンド（WGPU, CUDA, NdArray/CPU）を動的に切り替えられます。
  - **インタラクティブな学習UI**: `burn-train`に組み込まれたCUIダッシュボードにより、学習の進捗（損失、学習率など）をリアルタイムで確認できます。

-----

## PINNの仕組み

このモデルは、\*\*ニューラルネットワーク（探索者）**と**物理モデル（評価者）\*\*を組み合わせたハイブリッドアプローチです。学習プロセスは、CSVデータと物理計算を連携させることで機能します。

### 学習プロセスにおけるデータと物理モデルの連携

学習の一度のステップは、以下の流れで実行されます。

1.  **データ供給**: `fem_data_augmented.csv` からデータが一行読み込まれます。この行から、**目標周波数** ($f_{target}$) のみを取り出します。
2.  **モデルによる寸法予測**: ステップ1で取り出した**目標周波数** ($f_{target}$) をニューラルネットワークに入力します。ネットワークは、その周波数を実現するのに最適と思われる5つの寸法 ($\vec{d}_{pred}$) を予測（出力）します。
3.  **物理モデルによる周波数検証**: ステップ2で予測された寸法 ($\vec{d}*{pred}$) を、`physics.rs` 内の**片持ち梁の物理方程式**に代入します。 これにより、予測された寸法が実際にどの周波数 ($f*{pred}$) を生み出すかが理論的に計算されます。
4.  **損失の計算 (Mix Point)**: 物理モデルが算出した周波数 ($f_{pred}$) と、CSVから取り出した元の**目標周波数** ($f_{target}$) との間の誤差を計算します。 この誤差が「損失」となります。
5.  **パラメータ更新**: 損失が最小になるように、ニューラルネットワークの内部パラメータが更新されます。

このサイクルを繰り返すことで、ニューラルネットワークは「CSVの目標周波数」と「物理計算から得られる周波数」が一致するような寸法を予測する能力を身につけていきます。CSVの寸法データは直接の正解として使われるのではなく、物理法則を満たすべき目標値を提示する役割を担っています。

-----

## 数理的な詳細 (Mathematical Details)

### 1\. ニューラルネットワーク (Forward Pass)

このモデルは、入力層、2つの隠れ層、出力層からなる多層パーセプトロン（MLP）です。

  - **入力**: 目標周波数 $`f_{target}`$ (スカラー値)

  - **処理**:

    1.  **隠れ層1**: $`\vec{h_1} = \text{ReLU}(\text{LayerNorm}(\mathbf{W}_1 f_{target} + \vec{b}_1))`$
    2.  **隠れ層2**: $`\vec{h_2} = \text{ReLU}(\text{LayerNorm}(\mathbf{W}_2 \vec{h_1} + \vec{b}_2))`$
    3.  **出力層**: $`\vec{d}_{out} = \mathbf{W}_{out} \vec{h_2} + \vec{b}_{out}`$

  - **出力**: 各寸法が正の値となるようにsoftplus関数を適用した、5次元のベクトル $`\vec{d}_{prod}`$ は、

    $$
    \vec{d}_{pred} = \text{softplus}(\vec{d}_{out}) = \log(1 + \exp(\vec{d}_{out}))
    $$

    ここで
    
    $$
    \vec{d}_{pred} = [L_h, D_h, L_p, D_p, G_p]
    $$
    
    は、それぞれ柄の長さ、柄の直径、プロングの長さ、プロングの直径、プロングの間隔を表します。

### 2\. 物理モデル (Physics Model)

`src/physics.rs` の中核は、プロングを円形断面の片持ち梁と見なしたときの基本周波数を計算する以下の式です。

$$
f_{pred} = \frac{\lambda^2}{2\pi L_p^2} \sqrt{\frac{E \cdot I}{\rho \cdot A}}
$$

  - $`f_{pred}`$: 予測された寸法から計算される周波数
  - $`L_p`$: プロングの長さ (`PRONG_LENGTH`)
  - $`D_p`$: プロングの直径 (`PRONG_DIAMETER`)
  - $`A`$: プロングの断面積 ($`A = \frac{\pi}{4} D_p^2`$)
  - $`I`$: プロングの断面二次モーメント ($`I = \frac{\pi}{64} D_p^4`$)
  - $`\lambda^2`$: 1次曲げ振動モードの定数 (`K_FACTOR` = 3.5160)
  - $`E`$: ヤング率 (`YOUNGS_MODULUS`)
  - $`\rho`$: 密度 (`DENSITY`)

この式にモデルが予測した $L_p$ と $D_p$ を代入することで、$f_{pred}$ が計算されます。

### 3\. 損失関数 (Loss Function)

合計損失 $`\mathcal{L}_{total}`$ は、周波数損失とペナルティ損失の加重和として定義されます。

$$
\mathcal{L}_{total} = \text{mean} \left( \mathcal{L}_{freq} + \sum_{i} w_i \mathcal{L}_{penalty, i} \right)
$$

  - **周波数損失** $`\mathcal{L}_{freq}`$: 物理モデルの予測周波数と目標周波数の二乗相対誤差で、これが最小化の主要なターゲットです。
    $$
    \mathcal{L}_{freq} = \left( \frac{f_{pred} - f_{target}}{f_{target}} \right)^2
    $$
  - **ペナルティ損失** $`\mathcal{L}_{penalty, i}`$: 予測された寸法が物理的に妥当な範囲に収まるように制約を与えます。 例えば、寸法 $d$ が許容範囲 `[lower, upper]` から外れた場合のペナルティは以下のように計算されます。
    $$
    \mathcal{L}_{range} = \left( \frac{\text{ReLU}(\text{lower} - d)}{\text{lower}} \right)^2 + \left( \frac{\text{ReLU}(d - \text{upper})}{\text{upper}} \right)^2
    $$

-----

## プロジェクト構造

```
.
├── Cargo.toml
├── data/
│   └── fem_data_augmented.csv  # 学習用のデータセット
├── artifacts/                  # 学習済みモデルやログが保存されるディレクトリ
├── src/
│   ├── main.rs                 # アプリケーションのエントリポイント、CLI引数の解析
│   ├── constants.rs            # 物理定数やモデルの次元に関する定数
│   ├── dataset.rs              # CSVデータセットの読み込みとバッチ処理
│   ├── model.rs                # ニューラルネットワークモデルの定義
│   ├── physics.rs              # 物理法則に基づいた損失関数の定義
│   ├── train.rs                # モデルの学習処理
│   └── infer.rs                # 学習済みモデルを使った推論処理
└── README.md                   # このファイル
```

-----

## セットアップ

1.  **Rust Toolchain**: [公式サイト](https://www.rust-lang.org/tools/install)の手順に従って、最新のRustをインストールします。（このプロジェクトはRust 2024 Editionを使用しています）
2.  **CUDA (任意)**: CUDAバックエンドを使用する場合は、NVIDIAドライバと[CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)をインストールしてください。
3.  **依存関係のインストール**:
    ```bash
    cargo build --release
    ```

-----

## 使い方

### 学習

以下のコマンドでモデルの学習を開始します。`--backend`フラグで計算バックエンドを選択できます。

  - **WGPU (デフォルト)**
    ```bash
    cargo run --release -- train --backend wgpu
    ```
  - **CUDA**
    ```bash
    cargo run --release -- train --backend cuda
    ```
  - **CPU (NdArray)**
    ```bash
    cargo run --release -- train --backend nd-array
    ```

### 推論

学習が完了すると`artifacts`ディレクトリにモデルが保存されます。以下のコマンドで、指定した周波数に対する音叉の寸法を予測します。

```bash
# 例: 440Hzの音叉の寸法を予測
cargo run --release -- infer --freq 440.0

# CUDAバックエンドで推論する場合
cargo run --release -- --backend cuda infer --freq 440.0
```

