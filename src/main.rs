use aligned::{Aligned, A16};
use log::SetLoggerError;
use simple_logger::SimpleLogger;
use tfmicro::{AllOpResolver, MicroInterpreter, Model};

macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        if !($x - $y < $d || $y - $x < $d) {
            panic!();
        }
    };
}

#[derive(Debug)]
enum Error {
    TfMicro(tfmicro::Error),
    TfMicroStatus(tfmicro::Status),
    Log(SetLoggerError),
}

impl From<tfmicro::Error> for Error {
    fn from(e: tfmicro::Error) -> Self {
        Error::TfMicro(e)
    }
}

impl From<tfmicro::Status> for Error {
    fn from(e: tfmicro::Status) -> Self {
        Error::TfMicroStatus(e)
    }
}

impl From<SetLoggerError> for Error {
    fn from(e: SetLoggerError) -> Self {
        Error::Log(e)
    }
}

fn main() -> Result<(), Error> {
    SimpleLogger::new().init()?;

    let model_array = include_bytes!("../../models/2022-03-11-model.tfmicro");
    //let model = FlatBufferModel::build_from_model(&model)?;
    log::info!("Call Model::from_buffer");
    let model = Model::from_buffer(&model_array[..])?;

    const NUM_IN_FLOATS: usize = 5760;
    const NUM_OUT_FLOATS: usize = 3;
    const NUM_MARGIN_FLOATS: usize = 2000;
    const NUM_ACTIVATION_BUFFERS: usize = 4464;
    const TENSOR_ARENA_SIZE: usize =
        4 * (NUM_IN_FLOATS + NUM_OUT_FLOATS + NUM_MARGIN_FLOATS + NUM_ACTIVATION_BUFFERS);
    let mut arena: Aligned<A16, [u8; TENSOR_ARENA_SIZE]> = Aligned([0u8; TENSOR_ARENA_SIZE]);

    log::info!("Call AllOpResolver::new");
    let op_resolver = AllOpResolver::new();

    /*
    let builder = InterpreterBuilder::new(model, &resolver)?;
    let mut interpreter = builder.build()?;*/
    log::info!("Call MicroInterpreter::new");
    let mut interpreter = match MicroInterpreter::new(&model, op_resolver, &mut arena[..]) {
        Ok(i) => i,
        Err(e) => panic!("Error constructing interpreter:\n{:#?}", e),
    };

    // Allocate tensor buffers.
    //interpreter.allocate_tensors()?;

    /*let inputs = interpreter.inputs().to_vec();
    assert_eq!(inputs.len(), 1);*/

    //let input_index = inputs[0];

    /*let outputs = interpreter.outputs().to_vec();
    assert_eq!(outputs.len(), 1);*/

    //let output_index = outputs[0];

    /*let input_tensor = interpreter.tensor_info(input_index)?;
    assert_eq!(input_tensor.dims, vec![1, 1, 20, 9 * 16 * 2]);*/

    /*let output_tensor = interpreter.tensor_info(output_index)?;
    assert_eq!(output_tensor.dims, vec![1, 3]);*/

    // Generate input data
    /*
    let input_data = interpreter.tensor_data_mut(input_index)?;
    input_data.clone_from_slice(&vec![1f32; 20 * 9 * 16 * 2]);*/
    let input_data = [1f32; 20 * 9 * 16 * 2];
    interpreter.input(0, &input_data)?;

    // Run inference
    log::info!("Call MicroInterpreter::new");
    interpreter.invoke()?;

    // Read output buffers
    //let output: &[f32] = interpreter.tensor_data(output_index)?;
    let output: &[f32] = interpreter.output(0).as_data::<f32>();

    dbg!("Output: {:?}", output);

    let correct = &[0.4572307, 0.53414774, 0.];
    for (c, o) in correct.iter().zip(output) {
        assert_delta!(c, o, 0.01);
    }

    Ok(())
}
