#[derive(Debug)]
pub enum Error {
    TfMicro(tfmicro::Error),
    TfMicroStatus(tfmicro::Status),
    Log(log::SetLoggerError),
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

impl From<log::SetLoggerError> for Error {
    fn from(e: log::SetLoggerError) -> Self {
        Error::Log(e)
    }
}
