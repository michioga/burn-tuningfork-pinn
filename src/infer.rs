//! src/infer.rs


use crate::constants::model_dims;
use crate::model::TuningForkPINN;
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
};

pub fn run<B: Backend>(freq: f32, device: B::Device) {
    let artifact_dir = "./artifacts";
    let model_path = format!("{artifact_dir}/model");

    let record = CompactRecorder::new()
        .load(model_path.into(), &device)
        .expect("Failed to load model record. Run training first via `cargo run --release -- train`");

    let model: TuningForkPINN<B> = TuningForkPINN::new(&device).load_record(record);

    let input = Tensor::<B, 2>::from_floats([[freq]], &device);

    let dims = model.forward(input);
    let dims_values: Vec<f32> = dims.into_data().convert::<f32>().into_vec().unwrap();

    // ⭐️⭐️⭐️ 修正点: インデックスを定数に ⭐️⭐️⭐️
    println!("\n--- Predicted Dimensions (in meters) ---");
    println!("  - Handle Length:     {:.6}", dims_values[model_dims::HANDLE_LENGTH_IDX]);
    println!("  - Handle Diameter:   {:.6}", dims_values[model_dims::HANDLE_DIAMETER_IDX]);
    println!("  - Prong Length:      {:.6}", dims_values[model_dims::PRONG_LENGTH_IDX]);
    println!("  - Prong Diameter:    {:.6}", dims_values[model_dims::PRONG_DIAMETER_IDX]);
    println!("  - Prong Gap:         {:.6}", dims_values[model_dims::PRONG_GAP_IDX]);
    println!("----------------------------------------");
}