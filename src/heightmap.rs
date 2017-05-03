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

use common::{map_vec_pos, Plane};
use png;
use png::{ColorType, BitDepth};
use std::fs::File;

#[derive(Default)]
pub struct Heightmap {
    resolution: i32,
    bordered_resolution: i32,
    xp_heightmap: Box<[f32]>,
    xn_heightmap: Box<[f32]>,
    yp_heightmap: Box<[f32]>,
    yn_heightmap: Box<[f32]>,
    zp_heightmap: Box<[f32]>,
    zn_heightmap: Box<[f32]>,
}

fn load_heightmap(filename: &str) -> Result<(Box<[f32]>, u32), String> {
    let file = try!(File::open(filename).map_err(|_| "Failed to open file"));
    let decoder = png::Decoder::new(file);
    let (info, mut reader) = try!(decoder
                                      .read_info()
                                      .map_err(|e| format!("Failed to create decoder: {}", e)));

    if info.width != info.height {
        return Err(format!("Heightmap width and height not equal. W = {}, H = {}",
                           info.width,
                           info.height));
    }

    let mut img_data = vec![0; info.buffer_size()];
    let mut heightmap = vec![0.0; (info.width * info.height) as usize];

    try!(reader
             .next_frame(&mut img_data)
             .map_err(|e| format!("Failed to read image data: {}", e)));

    match (info.color_type, info.bit_depth) {
        (ColorType::Grayscale, BitDepth::Eight) => {
            let mut idx = 0;
            let mut img_idx = 0;
            while idx < img_data.len() {
                let value = img_data[img_idx];
                let value = (value as f32) / 255.0;
                heightmap[idx] = value;
                idx += 1;
                img_idx += 1;
            }
        }
        (ColorType::Grayscale, BitDepth::Sixteen) => {
            let mut idx = 0;
            let mut img_idx = 0;
            while idx < img_data.len() {
                let hi = (img_data[img_idx] as u16) << 8;
                let lo = img_data[img_idx + 1] as u16;
                let value = hi | lo;
                let value = (value as f32) / 65535.0;
                heightmap[idx] = value;
                idx += 1;
                img_idx += 2;
            }
        }
        (ColorType::RGB, BitDepth::Eight) => {
            let mut idx = 0;
            let mut img_idx = 0;
            while idx < img_data.len() {
                let r = (img_data[img_idx] as u32) << 16;
                let g = (img_data[img_idx + 1] as u32) << 8;
                let b = img_data[img_idx + 2] as u32;
                let value = r | g | b;
                let value = (value as f32) / 16777215.0;
                heightmap[idx] = value;
                idx += 1;
                img_idx += 3;
            }
        }
        _ => return Err("Unsupported image format".to_owned()),
    }

    Ok((heightmap.into_boxed_slice(), info.width))
}

fn cubic_interp(p0: f64, p1: f64, p2: f64, p3: f64, alpha: f64) -> f64 {
    let a = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    let b = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    let c = -0.5 * p0 + 0.5 * p2;
    let d = p1;

    a * alpha * alpha * alpha +
    b * alpha * alpha +
    c * alpha +
    d
}

fn bicubic_interp(p00: f64, p10: f64, p20: f64, p30: f64,
                  p01: f64, p11: f64, p21: f64, p31: f64,
                  p02: f64, p12: f64, p22: f64, p32: f64,
                  p03: f64, p13: f64, p23: f64, p33: f64,
                  x: f64, y: f64) -> f64 {
    let x0 = cubic_interp(p00, p01, p02, p03, y);
    let x1 = cubic_interp(p10, p11, p12, p13, y);
    let x2 = cubic_interp(p20, p21, p22, p23, y);
    let x3 = cubic_interp(p30, p31, p32, p33, y);
    cubic_interp(x0, x1, x2, x3, x)
}

impl Heightmap {
    pub fn load(prefix: &str) -> Heightmap {
        let xp_filename = [prefix, "xp.png"].concat();
        let xn_filename = [prefix, "xn.png"].concat();
        let yp_filename = [prefix, "yp.png"].concat();
        let yn_filename = [prefix, "yn.png"].concat();
        let zp_filename = [prefix, "zp.png"].concat();
        let zn_filename = [prefix, "zn.png"].concat();
        let xp_heightmap = load_heightmap(&xp_filename).expect("Failed to load XP heightmap");
        let xn_heightmap = load_heightmap(&xn_filename).expect("Failed to load XN heightmap");
        let yp_heightmap = load_heightmap(&yp_filename).expect("Failed to load YP heightmap");
        let yn_heightmap = load_heightmap(&yn_filename).expect("Failed to load YN heightmap");
        let zp_heightmap = load_heightmap(&zp_filename).expect("Failed to load ZP heightmap");
        let zn_heightmap = load_heightmap(&zn_filename).expect("Failed to load ZN heightmap");

        if xp_heightmap.1 != xn_heightmap.1 || xn_heightmap.1 != yp_heightmap.1 ||
           yp_heightmap.1 != yn_heightmap.1 || yn_heightmap.1 != zp_heightmap.1 ||
           zp_heightmap.1 != zn_heightmap.1 {
            panic!("Not all heightmap faces have the same resolution")
        }

        let resolution = xp_heightmap.1 as i32;
        Heightmap::new(&xp_heightmap.0,
                       &xn_heightmap.0,
                       &yp_heightmap.0,
                       &yn_heightmap.0,
                       &zp_heightmap.0,
                       &zn_heightmap.0,
                       resolution)
    }

    /// Creates a new `Heightmap` from the given heightmap data for each face.
    /// This type creates a border around each face's heightmap based from
    /// height data taken from adjacent faces, for use with bicubic
    /// interpolation.
    pub fn new(xp_heightmap: &[f32],
               xn_heightmap: &[f32],
               yp_heightmap: &[f32],
               yn_heightmap: &[f32],
               zp_heightmap: &[f32],
               zn_heightmap: &[f32],
               resolution: i32)
               -> Heightmap {
        let xp_bordered = Self::create_bordered_heightmap(Plane::XP,
                                                          xp_heightmap,
                                                          xn_heightmap,
                                                          yp_heightmap,
                                                          yn_heightmap,
                                                          zp_heightmap,
                                                          zn_heightmap,
                                                          resolution);
        let xn_bordered = Self::create_bordered_heightmap(Plane::XN,
                                                          xp_heightmap,
                                                          xn_heightmap,
                                                          yp_heightmap,
                                                          yn_heightmap,
                                                          zp_heightmap,
                                                          zn_heightmap,
                                                          resolution);
        let yp_bordered = Self::create_bordered_heightmap(Plane::YP,
                                                          xp_heightmap,
                                                          xn_heightmap,
                                                          yp_heightmap,
                                                          yn_heightmap,
                                                          zp_heightmap,
                                                          zn_heightmap,
                                                          resolution);
        let yn_bordered = Self::create_bordered_heightmap(Plane::YN,
                                                          xp_heightmap,
                                                          xn_heightmap,
                                                          yp_heightmap,
                                                          yn_heightmap,
                                                          zp_heightmap,
                                                          zn_heightmap,
                                                          resolution);
        let zp_bordered = Self::create_bordered_heightmap(Plane::ZP,
                                                          xp_heightmap,
                                                          xn_heightmap,
                                                          yp_heightmap,
                                                          yn_heightmap,
                                                          zp_heightmap,
                                                          zn_heightmap,
                                                          resolution);
        let zn_bordered = Self::create_bordered_heightmap(Plane::ZN,
                                                          xp_heightmap,
                                                          xn_heightmap,
                                                          yp_heightmap,
                                                          yn_heightmap,
                                                          zp_heightmap,
                                                          zn_heightmap,
                                                          resolution);

        Heightmap {
            resolution,
            bordered_resolution: resolution + 2,
            xp_heightmap: xp_bordered,
            xn_heightmap: xn_bordered,
            yp_heightmap: yp_bordered,
            yn_heightmap: yn_bordered,
            zp_heightmap: zp_bordered,
            zn_heightmap: zn_bordered,
        }
    }

    fn create_bordered_heightmap(plane: Plane,
                                 xp_heightmap: &[f32],
                                 xn_heightmap: &[f32],
                                 yp_heightmap: &[f32],
                                 yn_heightmap: &[f32],
                                 zp_heightmap: &[f32],
                                 zn_heightmap: &[f32],
                                 resolution: i32)
                                 -> Box<[f32]> {
        let (heightmap,
             north_plane, north_heightmap,
             south_plane, south_heightmap,
             east_plane, east_heightmap,
             west_plane, west_heightmap) = match plane {
            Plane::XP => (xp_heightmap,
                          Plane::YP, yp_heightmap,
                          Plane::YN, yn_heightmap,
                          Plane::ZN, zn_heightmap,
                          Plane::ZP, zp_heightmap),
            Plane::XN => (xn_heightmap,
                          Plane::YP, yp_heightmap,
                          Plane::YN, yn_heightmap,
                          Plane::ZP, zp_heightmap,
                          Plane::ZN, zn_heightmap),
            Plane::YP => (yp_heightmap,
                          Plane::ZN, zn_heightmap,
                          Plane::ZP, zp_heightmap,
                          Plane::XP, xp_heightmap,
                          Plane::XN, xn_heightmap),
            Plane::YN => (yn_heightmap,
                          Plane::ZP, zp_heightmap,
                          Plane::ZN, zn_heightmap,
                          Plane::XP, xp_heightmap,
                          Plane::XN, xn_heightmap),
            Plane::ZP => (zp_heightmap,
                          Plane::YP, yp_heightmap,
                          Plane::YN, yn_heightmap,
                          Plane::XP, xp_heightmap,
                          Plane::XN, xn_heightmap),
            Plane::ZN => (zn_heightmap,
                          Plane::YP, yp_heightmap,
                          Plane::YN, yn_heightmap,
                          Plane::XN, xn_heightmap,
                          Plane::XP, xp_heightmap),
        };

        let bordered_resolution = resolution + 2;
        let mut bordered = Vec::new();
        bordered.resize((bordered_resolution * bordered_resolution) as usize, 0.0);

        // North
        let (dir, base) = map_vec_pos((1, 0),
                                      (0, 1),
                                      resolution - 1,
                                      plane,
                                      north_plane);
        let mut idx = Self::calc_pos(1, bordered_resolution - 1, bordered_resolution);
        for i in 0..resolution {
            let (x, y) = (base.0 + dir.0 * i, base.1 + dir.1 * i);
            let src_idx = Self::calc_pos(x, y, resolution);
            bordered[idx] = north_heightmap[src_idx];
            idx += Self::calc_step(1, 0, bordered_resolution);
        }
        // South
        let (dir, base) = map_vec_pos((1, 0),
                                      (0, resolution - 2),
                                      resolution - 1,
                                      plane,
                                      south_plane);
        let mut idx = Self::calc_pos(1, 0, bordered_resolution);
        for i in 0..resolution {
            let (x, y) = (base.0 + dir.0 * i, base.1 + dir.1 * i);
            let src_idx = Self::calc_pos(x, y, resolution);
            bordered[idx] = south_heightmap[src_idx];
            idx += Self::calc_step(1, 0, bordered_resolution);
        }
        // East
        let (dir, base) = map_vec_pos((0, 1),
                                      (1, 0),
                                      resolution - 1,
                                      plane,
                                      east_plane);
        let mut idx = Self::calc_pos(bordered_resolution - 1, 1, bordered_resolution);
        for i in 0..resolution {
            let (x, y) = (base.0 + dir.0 * i, base.1 + dir.1 * i);
            let src_idx = Self::calc_pos(x, y, resolution);
            bordered[idx] = east_heightmap[src_idx];
            idx += Self::calc_step(0, 1, bordered_resolution);
        }
        // West
        let (dir, base) = map_vec_pos((0, 1),
                                      (resolution - 2, 0),
                                      resolution - 1,
                                      plane,
                                      west_plane);
        let mut idx = Self::calc_pos(0, 1, bordered_resolution);
        for i in 0..resolution {
            let (x, y) = (base.0 + dir.0 * i, base.1 + dir.1 * i);
            let src_idx = Self::calc_pos(x, y, resolution);
            bordered[idx] = west_heightmap[src_idx];
            idx += Self::calc_step(0, 1, bordered_resolution);
        }

        // Assign corners
        bordered[Self::calc_pos(0, 0, bordered_resolution)] =
            heightmap[Self::calc_pos(0, 0, resolution)];
        bordered[Self::calc_pos(bordered_resolution - 1, 0, bordered_resolution)] =
            heightmap[Self::calc_pos(resolution - 1, 0, resolution)];
        bordered[Self::calc_pos(0, bordered_resolution - 1, bordered_resolution)] =
            heightmap[Self::calc_pos(0, resolution - 1, resolution)];
        bordered[Self::calc_pos(bordered_resolution - 1, bordered_resolution - 1, bordered_resolution)] =
            heightmap[Self::calc_pos(resolution - 1, resolution - 1, resolution)];

        for y in 0..resolution {
            for x in 0..resolution {
                let src_idx = Self::calc_pos(x, y, resolution);
                let dst_idx = Self::calc_pos(x + 1, y + 1, bordered_resolution);
                bordered[dst_idx] = heightmap[src_idx];
            }
        }

        bordered.into_boxed_slice()
    }

    pub fn get_height_data(&self, plane: Plane, a: i32, b: i32) -> f32 {
        let a = a + 1;
        let b = b + 1;

        let i = Self::calc_pos(a, b, self.bordered_resolution);

        match plane {
            Plane::XP => self.xp_heightmap[i],
            Plane::XN => self.xn_heightmap[i],
            Plane::YP => self.yp_heightmap[i],
            Plane::YN => self.yn_heightmap[i],
            Plane::ZP => self.zp_heightmap[i],
            Plane::ZN => self.zn_heightmap[i],
        }
    }

    pub fn sample(&self, plane: Plane, x: f32, y: f32) -> f32 {
        let max = self.resolution - 1;
        let x = x * max as f32;
        let y = y * max as f32;

        let ix = x as i32;
        let iy = y as i32;

        let ix = if ix == max { ix - 1 } else { ix };
        let iy = if iy == max { iy - 1 } else { iy };

        let alpha_x = x - ix as f32;
        let alpha_y = y - iy as f32;

        let x0 = ix - 1;
        let x1 = ix;
        let x2 = ix + 1;
        let x3 = ix + 2;

        let y0 = iy - 1;
        let y1 = iy;
        let y2 = iy + 1;
        let y3 = iy + 2;

        let p00 = self.get_height_data(plane, x0, y0);
        let p10 = self.get_height_data(plane, x1, y0);
        let p20 = self.get_height_data(plane, x2, y0);
        let p30 = self.get_height_data(plane, x3, y0);

        let p01 = self.get_height_data(plane, x0, y1);
        let p11 = self.get_height_data(plane, x1, y1);
        let p21 = self.get_height_data(plane, x2, y1);
        let p31 = self.get_height_data(plane, x3, y1);

        let p02 = self.get_height_data(plane, x0, y2);
        let p12 = self.get_height_data(plane, x1, y2);
        let p22 = self.get_height_data(plane, x2, y2);
        let p32 = self.get_height_data(plane, x3, y2);

        let p03 = self.get_height_data(plane, x0, y3);
        let p13 = self.get_height_data(plane, x1, y3);
        let p23 = self.get_height_data(plane, x2, y3);
        let p33 = self.get_height_data(plane, x3, y3);

        bicubic_interp(p00 as f64, p10 as f64, p20 as f64, p30 as f64,
                       p01 as f64, p11 as f64, p21 as f64, p31 as f64,
                       p02 as f64, p12 as f64, p22 as f64, p32 as f64,
                       p03 as f64, p13 as f64, p23 as f64, p33 as f64,
                       alpha_x as f64, alpha_y as f64) as f32
    }

    /// Utility function for calculating an index from a position
    fn calc_pos(x: i32, y: i32, resolution: i32) -> usize {
        (x + (resolution - 1 - y) * resolution) as usize
    }

    /// Utility function for calculting an index step value from a position
    /// delta (`x`, `y`).
    fn calc_step(x: i32, y: i32, resolution: i32) -> usize {
        (x - y * resolution) as usize
    }
}
