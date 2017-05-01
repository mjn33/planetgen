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

use super::*;

#[test]
fn test_map_quad_pos() {
    assert_eq!(map_quad_pos(QuadPos::NorthWest, Plane::XP, Plane::XP), QuadPos::NorthWest);
    assert_eq!(map_quad_pos(QuadPos::NorthEast, Plane::XP, Plane::XP), QuadPos::NorthEast);
    assert_eq!(map_quad_pos(QuadPos::SouthWest, Plane::XP, Plane::XP), QuadPos::SouthWest);
    assert_eq!(map_quad_pos(QuadPos::SouthEast, Plane::XP, Plane::XP), QuadPos::SouthEast);

    assert_eq!(map_quad_pos(QuadPos::NorthWest, Plane::XP, Plane::ZN), QuadPos::NorthWest);
    assert_eq!(map_quad_pos(QuadPos::NorthEast, Plane::XP, Plane::ZN), QuadPos::NorthEast);
    assert_eq!(map_quad_pos(QuadPos::SouthWest, Plane::XP, Plane::ZN), QuadPos::SouthWest);
    assert_eq!(map_quad_pos(QuadPos::SouthEast, Plane::XP, Plane::ZN), QuadPos::SouthEast);

    assert_eq!(map_quad_pos(QuadPos::NorthWest, Plane::XP, Plane::YP), QuadPos::SouthWest);
    assert_eq!(map_quad_pos(QuadPos::NorthEast, Plane::XP, Plane::YP), QuadPos::NorthWest);
    // lower left
    assert_eq!(map_quad_pos(QuadPos::SouthEast, Plane::XP, Plane::YP), QuadPos::NorthEast);

    assert_eq!(map_quad_pos(QuadPos::SouthWest, Plane::XP, Plane::YP), QuadPos::SouthEast);
    assert_eq!(map_quad_pos(QuadPos::SouthWest, Plane::ZN, Plane::YP), QuadPos::NorthEast);
    assert_eq!(map_quad_pos(QuadPos::SouthWest, Plane::XN, Plane::YP), QuadPos::NorthWest);

    assert_eq!(map_quad_pos(QuadPos::NorthWest, Plane::XP, Plane::YN), QuadPos::NorthEast);
    assert_eq!(map_quad_pos(QuadPos::NorthWest, Plane::ZN, Plane::YN), QuadPos::SouthEast);
    assert_eq!(map_quad_pos(QuadPos::NorthWest, Plane::XN, Plane::YN), QuadPos::SouthWest);

    assert_eq!(map_quad_pos(QuadPos::SouthEast, Plane::ZN, Plane::YP), QuadPos::NorthWest);
}

#[test]
fn test_map_quad_side() {
    assert_eq!(map_quad_side(QuadSide::North, Plane::YP, Plane::ZP), QuadSide::North);
    assert_eq!(map_quad_side(QuadSide::West, Plane::YP, Plane::XP), QuadSide::North);
    assert_eq!(map_quad_side(QuadSide::South, Plane::YP, Plane::ZN), QuadSide::North);
    assert_eq!(map_quad_side(QuadSide::East, Plane::YP, Plane::XN), QuadSide::North);
}

#[test]
fn test_map_vec_pos() {
    assert_eq!(map_vec_pos((2, 0), (0, 8), 16, Plane::XP, Plane::YP),
               ((0, 2), (8, 0)));
    assert_eq!(map_vec_pos((2, 0), (0, 8), 16, Plane::ZP, Plane::XP),
               ((2, 0), (0, 8)));
    assert_eq!(map_vec_pos((2, 0), (8, 0), 16, Plane::ZN, Plane::YP),
               ((-2, 0), (8, 16)));
}
