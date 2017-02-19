use std;
use std::error;
use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// Indicates the operation failed since the underlying object was already
    /// destroyed.
    ObjectDestroyed,
    /// The specified child index was out of bounds
    BadChildIdx,
    /// Data given doesn't match size of buffer
    WrongBufferLength,
    /// Uniform name given not found in shader program
    BadUniformName,
    /// Other unspecified error.
    Other
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // For now just use error description, may change later with more enum
        // variants.
        write!(f, "{}", <Error as error::Error>::description(&self))
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::ObjectDestroyed => "Cannot perform operation on a destroyed object",
            Error::BadChildIdx => "Child index out of bounds",
            Error::WrongBufferLength => "Data given doesn't match size of buffer",
            Error::BadUniformName => "Uniform name given not found in shader program",
            Error::Other => "Unspecified error"
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;
