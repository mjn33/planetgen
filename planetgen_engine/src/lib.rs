//! # Core engine

#[cfg(test)]
#[macro_use]
extern crate approx;
extern crate cgmath;
extern crate gl;
extern crate num;
extern crate sdl2;

mod alloc;
pub mod error;
mod gl_util;
mod scene;
mod traits;

pub use self::scene::*;
