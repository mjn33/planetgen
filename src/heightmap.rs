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

impl Heightmap {
    /// Creates a new `Heightmap` from the given heightmap data for each face.
    /// This type creates a border around each face's heightmap based from
    /// height data taken from adjacent faces, for use with bicubic
    /// interpolation.
    pub fn new(xp_heightmap: &[f32], xn_heightmap: &[f32],
           yp_heightmap: &[f32], yn_heightmap: &[f32],
           zp_heightmap: &[f32], zn_heightmap: &[f32],
           resolution: i32) -> Heightmap {
        let xp_bordered = Self::create_bordered_heightmap(Plane::XP,
                                                          xp_heightmap, xn_heightmap,
                                                          yp_heightmap, yn_heightmap,
                                                          zp_heightmap, zn_heightmap,
                                                          resolution);
        let xn_bordered = Self::create_bordered_heightmap(Plane::XN,
                                                          xp_heightmap, xn_heightmap,
                                                          yp_heightmap, yn_heightmap,
                                                          zp_heightmap, zn_heightmap,
                                                          resolution);
        let yp_bordered = Self::create_bordered_heightmap(Plane::YP,
                                                          xp_heightmap, xn_heightmap,
                                                          yp_heightmap, yn_heightmap,
                                                          zp_heightmap, zn_heightmap,
                                                          resolution);
        let yn_bordered = Self::create_bordered_heightmap(Plane::YN,
                                                          xp_heightmap, xn_heightmap,
                                                          yp_heightmap, yn_heightmap,
                                                          zp_heightmap, zn_heightmap,
                                                          resolution);
        let zp_bordered = Self::create_bordered_heightmap(Plane::ZP,
                                                          xp_heightmap, xn_heightmap,
                                                          yp_heightmap, yn_heightmap,
                                                          zp_heightmap, zn_heightmap,
                                                          resolution);
        let zn_bordered = Self::create_bordered_heightmap(Plane::ZN,
                                                          xp_heightmap, xn_heightmap,
                                                          yp_heightmap, yn_heightmap,
                                                          zp_heightmap, zn_heightmap,
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
                                 xp_heightmap: &[f32], xn_heightmap: &[f32],
                                 yp_heightmap: &[f32], yn_heightmap: &[f32],
                                 zp_heightmap: &[f32], zn_heightmap: &[f32],
                                 resolution: i32) -> Box<[f32]> {

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
        let (dir, base) = map_vec_pos((1, 0), (0, 1), resolution - 1, plane, north_plane);
        let mut idx = Self::calc_pos(1, bordered_resolution - 1, bordered_resolution);
        for i in 0..resolution {
            let (x, y) = (base.0 + dir.0 * i, base.1 + dir.1 * i);
            let src_idx = Self::calc_pos(x, y, resolution);
            bordered[idx] = north_heightmap[src_idx];
            idx += Self::calc_step(1, 0, bordered_resolution);
        }
        // South
        let (dir, base) = map_vec_pos((1, 0), (0, resolution - 2), resolution - 1, plane, south_plane);
        let mut idx = Self::calc_pos(1, 0, bordered_resolution);
        for i in 0..resolution {
            let (x, y) = (base.0 + dir.0 * i, base.1 + dir.1 * i);
            let src_idx = Self::calc_pos(x, y, resolution);
            bordered[idx] = south_heightmap[src_idx];
            idx += Self::calc_step(1, 0, bordered_resolution);
        }
        // East
        let (dir, base) = map_vec_pos((0, 1), (1, 0), resolution - 1, plane, east_plane);
        let mut idx = Self::calc_pos(bordered_resolution - 1, 1, bordered_resolution);
        for i in 0..resolution {
            let (x, y) = (base.0 + dir.0 * i, base.1 + dir.1 * i);
            let src_idx = Self::calc_pos(x, y, resolution);
            bordered[idx] = east_heightmap[src_idx];
            idx += Self::calc_step(0, 1, bordered_resolution);
        }
        // West
        let (dir, base) = map_vec_pos((0, 1), (resolution - 2, 0), resolution - 1, plane, west_plane);
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

    pub fn resolution(&self) -> i32 {
        self.resolution
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
