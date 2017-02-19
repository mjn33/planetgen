#![feature(rc_raw)]

//! # Core engine

#[cfg(test)]
#[macro_use]
extern crate approx;
extern crate cgmath;
#[macro_use]
extern crate glium;
extern crate num;

pub mod error;
mod scene;

pub use self::scene::*;
