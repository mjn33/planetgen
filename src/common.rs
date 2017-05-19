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

use num;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Plane {
    /// +x
    XP,
    /// -x
    XN,
    /// +y
    YP,
    /// -y
    YN,
    /// +z
    ZP,
    /// -z
    ZN,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuadPos {
    /// This quad is a root quad.
    None,
    NorthWest,
    NorthEast,
    SouthWest,
    SouthEast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuadSide {
    North,
    South,
    East,
    West,
}

bitflags! {
    pub flags QuadSideFlags: u32 {
        const QUAD_SIDE_FLAGS_NONE = 0x00,
        const QUAD_SIDE_FLAGS_NORTH = 0x01,
        const QUAD_SIDE_FLAGS_SOUTH = 0x02,
        const QUAD_SIDE_FLAGS_EAST = 0x04,
        const QUAD_SIDE_FLAGS_WEST = 0x08,
        const QUAD_SIDE_FLAGS_ALL = 0x0F,
    }
}

/// Calculates three values that allow mapping coordinates / vectors from one
/// plane to another.
///
/// Returns a tuple of `(origin, x_vec, y_vec)`. `origin` are the coordinates of
/// what `src_plane` expects the origin to be, `x_vec` is the vector which
/// describes what `src_plane` expects a unit `x` vector to be, and
/// correspondingly for `y_vec`. `origin` coordinates range from `(0, 0)` to
/// `(1, 1)`.
///
/// See [`map_vec_pos`](fn.map_vec_pos.html) for an example of what this
/// function allows.
///
/// # Panics
///
/// Panics if the given `src_plane` and `dst_plane` don't neighbour each other.
pub fn calc_plane_mapping<T: num::Signed>(src_plane: Plane,
                                          dst_plane: Plane)
                                          -> ((T, T), (T, T), (T, T)) {
    use Plane::*;
    let one = || T::one();
    let zero = || T::zero();
    match (src_plane, dst_plane) {
        (XP, XN) | (XN, XP) | (YP, YN) | (YN, YP) | (ZP, ZN) | (ZN, ZP) => {
            // N/A as these planes do not neighbour each other on the quad
            // sphere.
            panic!("`src_plane` and `dst_plane` do not neighbour each other");
        }
        // YP
        (YP, XP) => ((zero(), one()), (zero(), -one()), (one(), zero())),
        (YP, XN) => ((one(), zero()), (zero(), one()), (-one(), zero())),
        (YP, ZN) => ((one(), one()), (-one(), zero()), (zero(), -one())),
        (XP, YP) => ((one(), zero()), (zero(), one()), (-one(), zero())),
        (XN, YP) => ((zero(), one()), (zero(), -one()), (one(), zero())),
        (ZN, YP) => ((one(), one()), (-one(), zero()), (zero(), -one())),
        // YN
        (YN, XP) => ((one(), zero()), (zero(), one()), (-one(), zero())),
        (YN, XN) => ((zero(), one()), (zero(), -one()), (one(), zero())),
        (YN, ZN) => ((one(), one()), (-one(), zero()), (zero(), -one())),
        (XP, YN) => ((zero(), one()), (zero(), -one()), (one(), zero())),
        (XN, YN) => ((one(), zero()), (zero(), one()), (-one(), zero())),
        (ZN, YN) => ((one(), one()), (-one(), zero()), (zero(), -one())),
        _ => ((zero(), zero()), (one(), zero()), (zero(), one())),
    }
}

pub fn map_quad_pos(pos: QuadPos, src_plane: Plane, dst_plane: Plane) -> QuadPos {
    let (x, y) = match pos {
        QuadPos::NorthWest => (0, 1),
        QuadPos::NorthEast => (1, 1),
        QuadPos::SouthWest => (0, 0),
        QuadPos::SouthEast => (1, 0),
        QuadPos::None => return QuadPos::None, // No other mapping applicable
    };

    let (origin, dir_x, dir_y) = calc_plane_mapping::<i32>(src_plane, dst_plane);
    let new_x = origin.0 + dir_x.0 * x + dir_y.0 * y;
    let new_y = origin.1 + dir_x.1 * x + dir_y.1 * y;

    match (new_x, new_y) {
        (0, 1) => QuadPos::NorthWest,
        (1, 1) => QuadPos::NorthEast,
        (0, 0) => QuadPos::SouthWest,
        (1, 0) => QuadPos::SouthEast,
        _ => unreachable!(),
    }
}

pub fn map_quad_side(side: QuadSide, src_plane: Plane, dst_plane: Plane) -> QuadSide {
    let (_, dir_x, dir_y) = calc_plane_mapping::<i32>(src_plane, dst_plane);
    let dir = match side {
        QuadSide::North => (dir_y.0, dir_y.1),
        QuadSide::South => (-dir_y.0, -dir_y.1),
        QuadSide::East => (dir_x.0, dir_x.1),
        QuadSide::West => (-dir_x.0, -dir_x.1),
    };

    match dir {
        (0, 1) => QuadSide::North,
        (0, -1) => QuadSide::South,
        (1, 0) => QuadSide::East,
        (-1, 0) => QuadSide::West,
        _ => unreachable!(),
    }
}

/// Maps a given vector and position on one plane to another. It allows other
/// code to deal with quads on differing planes as if they were on the same
/// plane, e.g.:
///
///     +--------+--------+
///     |        |        |
///     |   YP   |   XP   |
///     |        |        |
///     +--------+--------+
///
/// Say for example we want to iterate over coordinates bordering two quads on
/// these two different planes; if we were to ignore the fact the two quads are
/// on different planes we would start at `position = (MAX_COORD, 0)` and move
/// with `vector = (0, 1)` on YP and start at `position = (0, 0)` and move with
/// `vector = (0, 1)` on XP. However the position and vector for XP are
/// incorrect; this function allows those vectors for XP to be translated to the
/// correct values, in this case to `position = (0, MAX_COORD)` and `vector =
/// (1, 0)`.
///
/// # Parameters
///
///   * `vec` - The vector to map
///
///   * `pos` - The position to map
///
///   * `max_coord` - The maximum valid x or y coordinate value
///
///   * `src_plane` - The plane the given vector `vec` and position `pos` are
///     relative to
///
///   * `dst_plane` - The plane to map the given vector and position to
///
/// # Returns
///
/// A tuple containing the mapped vector first and the mapped position second.
pub fn map_vec_pos<T: num::Signed + Copy>(vec: (T, T),
                                          pos: (T, T),
                                          max_coord: T,
                                          src_plane: Plane,
                                          dst_plane: Plane)
                                          -> ((T, T), (T, T)) {
    let (origin, dir_x, dir_y) = calc_plane_mapping::<T>(src_plane, dst_plane);
    let mapped_vec = (vec.0 * dir_x.0 + vec.1 * dir_y.0, vec.0 * dir_x.1 + vec.1 * dir_y.1);
    let origin = (origin.0 * max_coord, origin.1 * max_coord);
    let mapped_pos = (origin.0 + dir_x.0 * pos.0 + dir_y.0 * pos.1,
                      origin.1 + dir_x.1 * pos.0 + dir_y.1 * pos.1);
    (mapped_vec, mapped_pos)
}

impl QuadPos {
    pub fn to_idx(&self) -> usize {
        match *self {
            QuadPos::None => panic!("Cannot convert QuadPos::None to an index!"),
            QuadPos::NorthWest => 0,
            QuadPos::NorthEast => 1,
            QuadPos::SouthWest => 2,
            QuadPos::SouthEast => 3,
        }
    }

    /// Splits the given `pos` into its vertical and horizontal side,
    /// e.g. upper-left maps to north and west.
    pub fn split(self) -> (QuadSide, QuadSide) {
        match self {
            QuadPos::NorthWest => (QuadSide::North, QuadSide::West),
            QuadPos::NorthEast => (QuadSide::North, QuadSide::East),
            QuadPos::SouthWest => (QuadSide::South, QuadSide::West),
            QuadPos::SouthEast => (QuadSide::South, QuadSide::East),
            QuadPos::None => panic!("Cannot call `split` on QuadPos::None."),
        }
    }

    /// Returns the position opposite to `pos`.
    pub fn opposite(self) -> QuadPos {
        match self {
            QuadPos::NorthWest => QuadPos::SouthEast,
            QuadPos::NorthEast => QuadPos::SouthWest,
            QuadPos::SouthWest => QuadPos::NorthEast,
            QuadPos::SouthEast => QuadPos::NorthWest,
            QuadPos::None => panic!("Cannot call `opposite` on QuadPos::None."),
        }
    }
}

impl From<QuadSide> for QuadSideFlags {
    fn from(side: QuadSide) -> Self {
        match side {
            QuadSide::North => QUAD_SIDE_FLAGS_NORTH,
            QuadSide::South => QUAD_SIDE_FLAGS_SOUTH,
            QuadSide::East => QUAD_SIDE_FLAGS_EAST,
            QuadSide::West => QUAD_SIDE_FLAGS_WEST,
        }
    }
}

impl QuadSideFlags {
    pub fn to_idx(&self) -> usize {
        self.bits() as usize
    }
}
