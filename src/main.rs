//! src/main.rs
#![recursion_limit = "256"]
#![allow(clippy::single_component_path_imports)]
// CudaとNdArrayは使っていないので削除
use burn::backend::{wgpu::Wgpu, Autodiff};
use clap::{Parser, Subcommand};

// 各モジュールをインポート
mod constants;
mod dataset;
mod infer;
mod model;
mod physics;
mod train;

#[derive(Parser, Debug)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// モデルを学習させる
    Train,
    /// 学習済みモデルで推論する
    Infer {
        /// 推奨を得たい周波数 (Hz)
        #[arg(short, long)]
        freq: f32,
    },
}

fn main() {
    let cli = Cli::parse();
    // デフォルトのバックエンドとしてWGPUを使用
    type MyBackend = Wgpu;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    match cli.command {
        Commands::Train => {
            train::run::<MyAutodiffBackend>(device);
        }
        Commands::Infer { freq } => {
            infer::run::<MyBackend>(freq, device);
        }
    }
}