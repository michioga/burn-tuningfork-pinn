//! src/main.rs
#![recursion_limit = "256"]
#![allow(clippy::single_component_path_imports)]

use burn::backend::{
    wgpu::{Wgpu, WgpuDevice},
    cuda::{Cuda, CudaDevice},
    ndarray::{NdArray, NdArrayDevice},
    Autodiff,
};
use burn::tensor::backend::AutodiffBackend;

use clap::{Parser, Subcommand, ValueEnum};

// Project modules remain the same
mod constants;
mod dataset;
mod infer;
mod model;
mod physics;
mod train;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, value_enum, default_value_t = BackendChoice::Wgpu)]
    backend: BackendChoice,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Train,
    Infer {
        #[arg(short, long)]
        freq: f32,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum BackendChoice {
    Wgpu,
    Cuda,
    NdArray,
}

fn run_app<B: AutodiffBackend>(cli: Cli, device: B::Device) {
    match cli.command {
        Commands::Train => {
            train::run::<B>(device);
        }
        Commands::Infer { freq } => {
            infer::run::<B::InnerBackend>(freq, device);
        }
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.backend {
        BackendChoice::Wgpu => {
            println!("🚀 Using WGPU backend...");
            type Backend = Autodiff<Wgpu>;
            let device = WgpuDevice::default();
            run_app::<Backend>(cli, device);
        }
        BackendChoice::Cuda => {
            println!("🚀 Using CUDA backend...");
            type Backend = Autodiff<Cuda>;
            let device = CudaDevice::default();
            run_app::<Backend>(cli, device);
        }
        BackendChoice::NdArray => {
            println!("🚀 Using NdArray (CPU) backend...");
            type Backend = Autodiff<NdArray<f32>>;
            let device = NdArrayDevice::default();
            run_app::<Backend>(cli, device);
        }
    }
}