bitflags! {
    pub flags PatchFlags: u32 {
        const PATCH_FLAGS_NONE = 0x00,
        const PATCH_FLAGS_NORTH = 0x01,
        const PATCH_FLAGS_SOUTH = 0x02,
        const PATCH_FLAGS_EAST = 0x04,
        const PATCH_FLAGS_WEST = 0x08,
    }
}

/// Utility function for computing the index of a vertex with a given stride.
pub fn vert_off(x: u16, y: u16, stride: u16) -> u16 {
    x * stride + y
}

/// Utility function for pushing a triangle to an indices buffer.
fn push_tri(indices: &mut Vec<u16>, a: u16, b: u16, c: u16) {
    indices.push(a); indices.push(b); indices.push(c);
}

/// Generate indices to fill the specified range on a grid of size `size`.
fn gen_indices_range(indices: &mut Vec<u16>, x1: u16, y1: u16, x2: u16, y2: u16, size: u16) {
    // TODO: maybe debug_assert! this instead?
    let (x1, x2) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
    let (y1, y2) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
    let adj_size = size + 1;
    let vert_off = |x, y| vert_off(x, y, adj_size);
    for x in x1..x2 {
        for y in y1..y2 {
            if ((x + y) % 2) == 1 {
                // *--------*
                // |-     2 |
                // |  -     |
                // |    -   |
                // | 1    - |
                // *--------*
                let t1a = vert_off(x, y);
                let t1b = vert_off(x + 1, y);
                let t1c = vert_off(x, y + 1);

                let t2a = vert_off(x + 1, y);
                let t2b = vert_off(x + 1, y + 1);
                let t2c = vert_off(x, y + 1);

                push_tri(indices, t1a, t1b, t1c);
                push_tri(indices, t2a, t2b, t2c);
            } else {
                // *--------*
                // | 2    - |
                // |    -   |
                // |  -     |
                // |-    1  |
                // *--------*
                let t1a = vert_off(x, y);
                let t1b = vert_off(x + 1, y);
                let t1c = vert_off(x + 1, y + 1);

                let t2a = vert_off(x + 1, y + 1);
                let t2b = vert_off(x, y + 1);
                let t2c = vert_off(x, y);

                push_tri(indices, t1a, t1b, t1c);
                push_tri(indices, t2a, t2b, t2c);
            }
        }
    }
}

/// Generate the indices for a quad of the given size and the specified
/// edges "patched".
pub fn gen_indices(size: u16, sides: PatchFlags) -> Vec<u16> {
    let adj_size = size + 1;
    let vert_off = |x, y| vert_off(x, y, adj_size);
    let mut indices = Vec::new();
    gen_indices_range(&mut indices, 1, 1, size - 1, size - 1, size);
    if sides.contains(PATCH_FLAGS_WEST) {
        for y in 1..adj_size-1 {
            if (y % 2) == 1 {
                // *
                // |-
                // |  -
                // |    -
                // |      -
                // |        *
                // |      -
                // |    -
                // |  -
                // |-
                // *
                let ta = vert_off(0, y - 1);
                let tb = vert_off(1, y);
                let tc = vert_off(0, y + 1);

                push_tri(&mut indices, ta, tb, tc);
            } else {
                //          *
                //         -|
                //       -  |
                //     -  2 |
                //   -      |
                // *--------*
                //   -      |
                //     -  1 |
                //       -  |
                //         -|
                //          *
                let t1a = vert_off(1, y - 1);
                let t1b = vert_off(1, y);
                let t1c = vert_off(0, y);

                let t2a = vert_off(0, y);
                let t2b = vert_off(1, y);
                let t2c = vert_off(1, y + 1);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            }
        }
    } else {
        for y in 1..adj_size-1 {
            if (y % 2) == 1 {
                // *
                // |-
                // |  -
                // | 2  -
                // |      -
                // *--------*
                // |      -
                // | 1  -
                // |  -
                // |-
                // *
                let t1a = vert_off(1, y);
                let t1b = vert_off(0, y);
                let t1c = vert_off(0, y - 1);

                let t2a = vert_off(0, y + 1);
                let t2b = vert_off(0, y);
                let t2c = vert_off(1, y);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            } else {
                //          *
                //         -|
                //       -  |
                //     -  2 |
                //   -      |
                // *--------*
                //   -      |
                //     -  1 |
                //       -  |
                //         -|
                //          *
                let t1a = vert_off(1, y - 1);
                let t1b = vert_off(1, y);
                let t1c = vert_off(0, y);

                let t2a = vert_off(0, y);
                let t2b = vert_off(1, y);
                let t2c = vert_off(1, y + 1);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            }
        }
    }

    if sides.contains(PATCH_FLAGS_SOUTH) {
        for x in 1..adj_size-1 {
            if (x % 2) == 1 {
                //
                //          *
                //        -   -
                //      -       -
                //    -           -
                //  -               -
                // *-----------------*
                let ta = vert_off(x + 1, 0);
                let tb = vert_off(x, 1);
                let tc = vert_off(x - 1, 0);

                push_tri(&mut indices, ta, tb, tc);
            } else {
                // *--------*--------*
                //  -       |       -
                //    - 1   |  2  -
                //      -   |   -
                //        - | -
                //          *
                let t1a = vert_off(x, 0);
                let t1b = vert_off(x, 1);
                let t1c = vert_off(x - 1, 1);

                let t2a = vert_off(x + 1, 1);
                let t2b = vert_off(x, 1);
                let t2c = vert_off(x, 0);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            }
        }
    } else {
        for x in 1..adj_size-1 {
            if (x % 2) == 1 {
                //          *
                //        - | -
                //      -   |   -
                //    -  1  |  2  -
                //  -       |       -
                // *--------*--------*
                let t1a = vert_off(x - 1, 0);
                let t1b = vert_off(x, 0);
                let t1c = vert_off(x, 1);

                let t2a = vert_off(x, 1);
                let t2b = vert_off(x, 0);
                let t2c = vert_off(x + 1, 0);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            } else {
                // *--------*--------*
                //  -       |       -
                //    - 1   |  2  -
                //      -   |   -
                //        - | -
                //          *
                let t1a = vert_off(x, 0);
                let t1b = vert_off(x, 1);
                let t1c = vert_off(x - 1, 1);

                let t2a = vert_off(x + 1, 1);
                let t2b = vert_off(x, 1);
                let t2c = vert_off(x, 0);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            }
        }
    }

    if sides.contains(PATCH_FLAGS_EAST) {
        for y in 1..adj_size-1 {
            if (y % 2) == 1 {
                //          *
                //         -|
                //       -  |
                //     -    |
                //   -      |
                // *        |
                //   -      |
                //     -    |
                //       -  |
                //         -|
                //          *
                let ta = vert_off(adj_size - 1, y + 1);
                let tb = vert_off(adj_size - 2, y);
                let tc = vert_off(adj_size - 1, y - 1);

                push_tri(&mut indices, ta, tb, tc);
            } else {
                // *
                // |-
                // |  -
                // |  2 -
                // |      -
                // *--------*
                // |      -
                // |  1 -
                // |  -
                // |-
                // *
                let t1a = vert_off(adj_size - 1, y);
                let t1b = vert_off(adj_size - 2, y);
                let t1c = vert_off(adj_size - 2, y - 1);

                let t2a = vert_off(adj_size - 2, y + 1);
                let t2b = vert_off(adj_size - 2, y);
                let t2c = vert_off(adj_size - 1, y);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            }
        }
    } else {
        for y in 1..adj_size-1 {
            if (y % 2) == 1 {
                //          *
                //         -|
                //       -  |
                //     - 2  |
                //   -      |
                // *--------*
                //   -      |
                //     - 1  |
                //       -  |
                //         -|
                //          *
                let t1a = vert_off(adj_size - 1, y - 1);
                let t1b = vert_off(adj_size - 1, y);
                let t1c = vert_off(adj_size - 2, y);

                let t2a = vert_off(adj_size - 2, y);
                let t2b = vert_off(adj_size - 1, y);
                let t2c = vert_off(adj_size - 1, y + 1);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            } else {
                // *
                // |-
                // |  -
                // |  2 -
                // |      -
                // *--------*
                // |      -
                // |  1 -
                // |  -
                // |-
                // *
                let t1a = vert_off(adj_size - 1, y);
                let t1b = vert_off(adj_size - 2, y);
                let t1c = vert_off(adj_size - 2, y - 1);

                let t2a = vert_off(adj_size - 2, y + 1);
                let t2b = vert_off(adj_size - 2, y);
                let t2c = vert_off(adj_size - 1, y);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            }
        }
    }

    if sides.contains(PATCH_FLAGS_NORTH) {
        for x in 1..adj_size-1 {
            if (x % 2) == 1 {
                // *-----------------*
                //  -               -
                //    -           -
                //      -       -
                //        -   -
                //          *

                let ta = vert_off(x - 1, size);
                let tb = vert_off(x, size - 1);
                let tc = vert_off(x + 1, size);

                push_tri(&mut indices, ta, tb, tc);
            } else {
                //          *
                //        - | -
                //      -   |   -
                //    - 1   |   2 -
                //  -       |       -
                // *--------*--------*

                let t1a = vert_off(x - 1, size - 1);
                let t1b = vert_off(x, size - 1);
                let t1c = vert_off(x, size);

                let t2a = vert_off(x, size);
                let t2b = vert_off(x, size - 1);
                let t2c = vert_off(x + 1, size - 1);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            }
        }
    } else {
        for x in 1..adj_size-1 {
            if (x % 2) == 1 {
                // *--------*--------*
                //  -       |       -
                //    - 1   |   2 -
                //      -   |   -
                //        - | -
                //          *

                let t1a = vert_off(x, size - 1);
                let t1b = vert_off(x, size);
                let t1c = vert_off(x - 1, size);

                let t2a = vert_off(x + 1, size);
                let t2b = vert_off(x, size);
                let t2c = vert_off(x, size - 1);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            } else {
                //          *
                //        - | -
                //      -   |   -
                //    - 1   |   2 -
                //  -       |       -
                // *--------*--------*

                let t1a = vert_off(x - 1, size - 1);
                let t1b = vert_off(x, size - 1);
                let t1c = vert_off(x, size);

                let t2a = vert_off(x, size);
                let t2b = vert_off(x, size - 1);
                let t2c = vert_off(x + 1, size - 1);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            }
        }
    }

    indices
}
