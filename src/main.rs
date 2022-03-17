use tflite::model::Model;
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder, Result};

fn main() -> Result<()> {
    let model = Model::from_file("../models/2022-03-11-model.tflite")?;
    let model = FlatBufferModel::build_from_model(&model)?;

    let resolver = BuiltinOpResolver::default();

    let builder = InterpreterBuilder::new(model, &resolver)?;
    let mut interpreter = builder.build()?;

    // Allocate tensor buffers.
    interpreter.allocate_tensors()?;

    let inputs = interpreter.inputs().to_vec();
    assert_eq!(inputs.len(), 1);

    let input_index = inputs[0];

    let outputs = interpreter.outputs().to_vec();
    assert_eq!(outputs.len(), 1);

    let output_index = outputs[0];

    let input_tensor = interpreter.tensor_info(input_index).unwrap();
    assert_eq!(input_tensor.dims, vec![1, 1, 20, 9 * 16 * 2]);

    let output_tensor = interpreter.tensor_info(output_index).unwrap();
    assert_eq!(output_tensor.dims, vec![1, 3]);

    // Generate input data
    let input_data = interpreter.tensor_data_mut(input_index)?;
    input_data.clone_from_slice(&vec![1f32; 20 * 9 * 16 * 2]);

    // Run inference
    interpreter.invoke()?;

    // Read output buffers
    let output: &[f32] = interpreter.tensor_data(output_index)?;

    println!("Output: {:?}", output);

    Ok(())
}
