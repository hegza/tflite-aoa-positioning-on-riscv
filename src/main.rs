#![no_std]
#![no_main]

mod error;

use aligned::{Aligned, A16};
use core::{fmt::Write, panic::PanicInfo};
use riscv_rt::entry;
use tfmicro::{AllOpResolver, MicroInterpreter, Model};
use uart_16550::MmioSerialPort;

// Macro to assert that a float value is within given bounds
macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        if !($x - $y < $d || $y - $x < $d) {
            panic!();
        }
    };
}

// UART address
const SERIAL_PORT_BASE_ADDRESS: usize = 0x6000_1800;

/// Wrapper for the UART resource
struct GlobalSerial(Option<MmioSerialPort>);

// Global resource for the UART serial, zero init
static mut SERIAL_PORT: GlobalSerial = GlobalSerial(None);

#[entry]
unsafe fn main() -> ! {
    let mut serial_port = MmioSerialPort::new(SERIAL_PORT_BASE_ADDRESS);
    serial_port.init();

    SERIAL_PORT.0 = Some(serial_port);

    writeln!(SERIAL_PORT, "Start").unwrap();

    let model_array = include_bytes!("../../models/2022-03-11-model.tfmicro");
    let model = Model::from_buffer(&model_array[..]).unwrap();

    const NUM_IN_FLOATS: usize = 5760;
    const NUM_OUT_FLOATS: usize = 3;
    const NUM_MARGIN_FLOATS: usize = 2000;
    const NUM_ACTIVATION_BUFFERS: usize = 4464;
    const TENSOR_ARENA_SIZE: usize =
        4 * (NUM_IN_FLOATS + NUM_OUT_FLOATS + NUM_MARGIN_FLOATS + NUM_ACTIVATION_BUFFERS);
    let mut arena: Aligned<A16, [u8; TENSOR_ARENA_SIZE]> = Aligned([0u8; TENSOR_ARENA_SIZE]);

    let op_resolver = AllOpResolver::new();

    let mut interpreter = match MicroInterpreter::new(&model, op_resolver, &mut arena[..]) {
        Ok(i) => i,
        Err(e) => panic!("Error constructing interpreter:\n{:#?}", e),
    };

    // Generate input data
    let input_data = [1f32; 20 * 9 * 16 * 2];

    interpreter.input(0, &input_data).unwrap();

    // Run inference
    writeln!(SERIAL_PORT, "MicroInterpreter::invoke").unwrap();
    interpreter.invoke().unwrap();

    // Read output buffers
    let output: &[f32] = interpreter.output(0).as_data::<f32>();

    writeln!(SERIAL_PORT, "Output: {:?}", output).unwrap();

    let correct = &[0.4572307, 0.53414774, 0.];
    for (c, o) in correct.iter().zip(output) {
        assert_delta!(c, o, 0.01);
    }
    writeln!(SERIAL_PORT, "Assert OK!").unwrap();

    loop {}
}

// Print wrapper for UART
impl core::fmt::Write for GlobalSerial {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.0
            .as_mut()
            .map(|serial_port| write!(serial_port, "{}", s))
            .unwrap()
    }
}

// Write error to UART on panic
#[panic_handler]
unsafe fn panic(info: &PanicInfo) -> ! {
    writeln!(SERIAL_PORT, "{}", info).ok();

    loop {}
}
