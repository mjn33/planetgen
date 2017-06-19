// Copyright (c) 2017 Matthew J. Nicholls
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use std;
use std::error;
use std::fmt;

#[derive(Debug, Eq, PartialEq)]
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
