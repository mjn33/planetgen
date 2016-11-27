#![feature(rc_raw)]

//! # Core engine

#[macro_use]
extern crate approx;

extern crate cgmath;
#[macro_use]
extern crate glium;
extern crate num;

mod scene;

pub use self::scene::*;
