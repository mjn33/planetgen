#[macro_use]
extern crate bitflags;
extern crate cgmath;
extern crate glium;
extern crate num;
extern crate planetgen_engine;

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use cgmath::{Deg, Euler, InnerSpace, Rotation, Quaternion, Vector3};

use glium::DisplayBuild;

use num::One;

use planetgen_engine::{Behaviour, BehaviourMessages, Material, Mesh, Object, Scene, Shader, Vertex};

/// Generate a `Vec` containing a `size + 1` by `size + 1` grid of vertices.
fn gen_vertices(size: u16) -> Vec<Vertex> {
    let adj_size = size + 1;
    let mut vertices = Vec::new();
    let diff = 2.0 / size as f32;
    for x in 0..adj_size {
        for y in 0..adj_size {
            vertices.push(Vertex { vert_pos: [-1.0 + x as f32 * diff, -1.0 + y as f32 * diff, 0.0] });
        }
    }
    vertices
}

/// Utility function for computing the index of a vertex with a given stride.
fn vert_off(x: u16, y: u16, stride: u16) -> u16 {
    x * stride + y
}

/// Utility function for pushing a triangle to an indices buffer.
fn push_tri(indices: &mut Vec<u16>, a: u16, b: u16, c: u16) {
    indices.push(a); indices.push(b); indices.push(c);
}

/// Generate indices to fill the specified range on a grid of size `size`.
fn gen_indices_range(indices: &mut Vec<u16>, x1: u16, y1: u16, x2: u16, y2: u16, size: u16)  {
    // TODO: maybe debug_assert! this instead?
    let (x1, x2) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
    let (y1, y2) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
    let adj_size = size + 1;
    let vert_off = |x, y| vert_off(x, y, adj_size);
    for x in x1..x2 {
        for y in y1..y2 {
            // *--------*
            // | 2    - |
            // |    -   |
            // |  -     |
            // |-    1  |
            // *--------*
            let t1a = vert_off(x, y);
            let t1b = vert_off(x + 1, y);
            let t1c = vert_off(x, y + 1);

            let t2a = vert_off(x + 1, y);
            let t2b = vert_off(x + 1, y + 1);
            let t2c = vert_off(x, y + 1);

            push_tri(indices, t1a, t1b, t1c);
            push_tri(indices, t2a, t2b, t2c);
        }
    }
}

bitflags! {
    flags PatchSide: u32 {
        const PATCH_SIDE_NONE   = 0x00,
        const PATCH_SIDE_LEFT   = 0x01,
        const PATCH_SIDE_RIGHT  = 0x02,
        const PATCH_SIDE_TOP    = 0x04,
        const PATCH_SIDE_BOTTOM = 0x08,
    }
}

/// Generate the indices for a quad of the given size and the specified
/// edges "patched".
fn gen_indices(size: u16, sides: PatchSide) -> Vec<u16> {
    let adj_size = size + 1;
    let vert_off = |x, y| vert_off(x, y, adj_size);
    let mut indices = Vec::new();
    gen_indices_range(&mut indices, 1, 1, size - 1, size - 1, size);
    if sides.contains(PATCH_SIDE_LEFT) {
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
        if sides.contains(PATCH_SIDE_BOTTOM) {
            let ta = vert_off(1, 1);
            let tb = vert_off(0, 1);
            let tc = vert_off(0, 0);

            push_tri(&mut indices, ta, tb, tc);
        } else {
            gen_indices_range(&mut indices, 0, 0, 1, 1, size);
        }
        gen_indices_range(&mut indices, 0, 1, 1, size - 1, size);
        if sides.contains(PATCH_SIDE_TOP) {
            let ta = vert_off(0, size);
            let tb = vert_off(0, size - 1);
            let tc = vert_off(1, size - 1);

            push_tri(&mut indices, ta, tb, tc);
        } else {
            gen_indices_range(&mut indices, 0, size - 1, 1, size, size);
        }
    }

    if sides.contains(PATCH_SIDE_TOP) {
        for x in 1..adj_size-1 {
            if (x % 2) == 1 {
                let ta = vert_off(x - 1, size);
                let tb = vert_off(x, size - 1);
                let tc = vert_off(x + 1, size);

                push_tri(&mut indices, ta, tb, tc);
            } else {
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
        if sides.contains(PATCH_SIDE_LEFT) {
            let ta = vert_off(1, size - 1);
            let tb = vert_off(1, size);
            let tc = vert_off(0, size);

            push_tri(&mut indices, ta, tb, tc);
        }
        gen_indices_range(&mut indices, 1, size - 1, size - 1, size, size);
        if sides.contains(PATCH_SIDE_RIGHT) {
            let ta = vert_off(size, size);
            let tb = vert_off(size - 1, size);
            let tc = vert_off(size - 1, size - 1);

            push_tri(&mut indices, ta, tb, tc);
        }
    }

    if sides.contains(PATCH_SIDE_BOTTOM) {
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
        if sides.contains(PATCH_SIDE_LEFT) {
            let ta = vert_off(0, 0);
            let tb = vert_off(1, 0);
            let tc = vert_off(1, 1);

            push_tri(&mut indices, ta, tb, tc);
        }
        gen_indices_range(&mut indices, 1, 0, size - 1, 1, size);
        if sides.contains(PATCH_SIDE_RIGHT) {
            let ta = vert_off(size - 1, 1);
            let tb = vert_off(size - 1, 0);
            let tc = vert_off(size, 0);

            push_tri(&mut indices, ta, tb, tc);
        }
    }

    if sides.contains(PATCH_SIDE_RIGHT) {
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
                let t1a = vert_off(adj_size - 2, y);
                let t1b = vert_off(adj_size - 2, y - 1);
                let t1c = vert_off(adj_size - 1, y);

                let t2a = vert_off(adj_size - 2, y + 1);
                let t2b = vert_off(adj_size - 2, y);
                let t2c = vert_off(adj_size - 1, y);

                push_tri(&mut indices, t1a, t1b, t1c);
                push_tri(&mut indices, t2a, t2b, t2c);
            }
        }
    } else {
        if sides.contains(PATCH_SIDE_BOTTOM) {
            let ta = vert_off(size, 0);
            let tb = vert_off(size, 1);
            let tc = vert_off(size - 1, 1);

            push_tri(&mut indices, ta, tb, tc);
        } else {
            gen_indices_range(&mut indices, size - 1, 0, size, 1, size);
        }
        gen_indices_range(&mut indices, size - 1, 1, size, size - 1, size);
        if sides.contains(PATCH_SIDE_TOP) {
            let ta = vert_off(size - 1, size - 1);
            let tb = vert_off(size, size - 1);
            let tc = vert_off(size, size);

            push_tri(&mut indices, ta, tb, tc);
        } else {
            gen_indices_range(&mut indices, size - 1, size - 1, size, size, size);
        }
    }
    indices
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Plane {
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
enum QuadPos {
    /// This quad is a root quad.
    None,
    UpperLeft,
    UpperRight,
    LowerLeft,
    LowerRight,
}

impl QuadPos {
    fn to_idx(&self) -> usize {
        match *self {
            QuadPos::None => panic!("Cannot convert QuadPos::None to an index!"),
            QuadPos::UpperLeft => 0,
            QuadPos::UpperRight => 1,
            QuadPos::LowerLeft => 2,
            QuadPos::LowerRight => 3,
        }
    }
}

#[derive(Clone, Copy)]
struct VertCoord(Plane, u32, u32);

struct Quad {
    behaviour: Behaviour,
    plane: Plane,
    pos: QuadPos,
    mesh: Option<Rc<Mesh>>,
    shader: Option<Rc<Shader>>,
    material: Option<Rc<Material>>,
    base_coord: (u32, u32),
    cur_subdivision: u32,
    children: Option<[Rc<RefCell<Quad>>; 4]>,
    north: Option<Weak<RefCell<Quad>>>,
    south: Option<Weak<RefCell<Quad>>>,
    east: Option<Weak<RefCell<Quad>>>,
    west: Option<Weak<RefCell<Quad>>>,
}

impl Quad {
    fn init(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        let vert_step = 1 << (sphere.max_subdivision - self.cur_subdivision);
        let adj_size = sphere.quad_mesh_size + 1;
        let mut vertices = gen_vertices(sphere.quad_mesh_size);
        let indices = gen_indices(sphere.quad_mesh_size, PATCH_SIDE_NONE);

        let vert_off = |x, y| vert_off(x, y, adj_size);
        for x in 0..adj_size {
            for y in 0..adj_size {
                let vert_coord = VertCoord(self.plane,
                                           self.base_coord.0 + x as u32 * vert_step,
                                           self.base_coord.1 + y as u32 * vert_step);

                let vert_pos = map_vertcoord(vert_coord, sphere.max_coord).normalize();
                let off = vert_off(x, y);
                vertices[off as usize].vert_pos = [vert_pos.x, vert_pos.y, vert_pos.z];
            }
        }

        let mesh = scene.create_mesh(&vertices, &indices);
        let shader = scene.create_shader(
            include_str!("default_vs.glsl"),
            include_str!("default_fs.glsl"),
            None);
        let material = scene.create_material(shader.clone()).unwrap();

        self.mesh = Some(mesh);
        self.shader = Some(shader);
        self.material = Some(material);

        // TODO: reduce cloning
        let self_object = self.behaviour.object(scene).unwrap().clone();
        let sphere_object = sphere.behaviour().object(scene).unwrap().clone();
        scene.set_object_parent(&self_object, Some(&sphere_object));
    }

    fn mid_coord_pos(&self, sphere: &QuadSphere) -> Vector3<f32> {
        let half_quad_length = sphere.quad_length(self.cur_subdivision) / 2;
        let mid_coord = (self.base_coord.0 + half_quad_length, self.base_coord.1 + half_quad_length);
        map_vertcoord(VertCoord(self.plane, mid_coord.0, mid_coord.1), sphere.max_coord)
    }

    fn in_subdivision_range(&self, sphere: &QuadSphere) -> bool {
        if self.cur_subdivision == sphere.max_subdivision {
            return false
        }

        let extra = (1.05 as f32).powf((sphere.max_subdivision - self.cur_subdivision) as f32);
        let mid_coord_pos = self.mid_coord_pos(sphere).normalize();

        let centre_pos = sphere.centre_pos().normalize();
        let theta = centre_pos.dot(mid_coord_pos).acos();
        let r = 1.0;
        let dist = r * theta;

        // Quad length changes when deformed into sphere
        let pi_div_4 = std::f32::consts::PI / 4.0;
        let quad_length = sphere.quad_length(self.cur_subdivision);
        let real_quad_length = 2.0 * (quad_length as f32 / sphere.max_coord as f32) * pi_div_4;
        let range = extra * 1.5 * real_quad_length;
        dist <= range
    }

    fn in_collapse_range(&self, sphere: &QuadSphere) -> bool {
        let extra = (1.05 as f32).powf((sphere.max_subdivision - self.cur_subdivision) as f32);
        let mid_coord_pos = self.mid_coord_pos(sphere).normalize();

        let centre_pos = sphere.centre_pos();
        let theta = centre_pos.dot(mid_coord_pos).acos();
        let r = 1.0;
        let dist = r * theta;

        // Quad length changes when deformed into sphere
        let pi_div_4 = std::f32::consts::PI / 4.0;
        let quad_length = sphere.quad_length(self.cur_subdivision);
        let real_quad_length = 2.0 * (quad_length as f32 / sphere.max_coord as f32) * pi_div_4;
        let range = extra * 2.0 * 1.5 * real_quad_length;
        dist >= range
    }

    fn get_child(&self, pos: QuadPos) -> Option<&Rc<RefCell<Quad>>> {
        self.children.as_ref().map(|children| &children[pos.to_idx()])
    }

    fn direct_north(&self) -> Option<Weak<RefCell<Quad>>> {
        match self.pos {
            QuadPos::LowerLeft | QuadPos::LowerRight => self.north.clone(),
            QuadPos::UpperLeft => {
                let north = self.north.as_ref().unwrap().upgrade().unwrap();
                let north_borrow = north.borrow();
                Some(Rc::downgrade(north_borrow.get_child(QuadPos::LowerLeft).unwrap()))
            },
            QuadPos::UpperRight => {
                let north = self.north.as_ref().unwrap().upgrade().unwrap();
                let north_borrow = north.borrow();
                Some(Rc::downgrade(north_borrow.get_child(QuadPos::LowerRight).unwrap()))
            },
            QuadPos::None => self.north.clone(),
        }
    }

    fn direct_south(&self) -> Option<Weak<RefCell<Quad>>> {
        match self.pos {
            QuadPos::UpperLeft | QuadPos::UpperRight => self.south.clone(),
            QuadPos::LowerLeft => {
                let south = self.south.as_ref().unwrap().upgrade().unwrap();
                let south_borrow = south.borrow();
                Some(Rc::downgrade(south_borrow.get_child(QuadPos::UpperLeft).unwrap()))
            },
            QuadPos::LowerRight => {
                let south = self.south.as_ref().unwrap().upgrade().unwrap();
                let south_borrow = south.borrow();
                Some(Rc::downgrade(south_borrow.get_child(QuadPos::UpperRight).unwrap()))
            },
            QuadPos::None => self.south.clone(),
        }
    }

    fn direct_east(&self) -> Option<Weak<RefCell<Quad>>> {
        match self.pos {
            QuadPos::UpperLeft | QuadPos::LowerLeft => self.east.clone(),
            QuadPos::UpperRight => {
                let east = self.east.as_ref().unwrap().upgrade().unwrap();
                let east_borrow = east.borrow();
                Some(Rc::downgrade(east_borrow.get_child(QuadPos::UpperLeft).unwrap()))
            },
            QuadPos::LowerRight => {
                let east = self.east.as_ref().unwrap().upgrade().unwrap();
                let east_borrow = east.borrow();
                Some(Rc::downgrade(east_borrow.get_child(QuadPos::LowerLeft).unwrap()))
            },
            QuadPos::None => self.east.clone(),
        }
    }

    fn direct_west(&self) -> Option<Weak<RefCell<Quad>>> {
        match self.pos {
            QuadPos::UpperRight | QuadPos::LowerRight => self.west.clone(),
            QuadPos::UpperLeft => {
                let west = self.west.as_ref().unwrap().upgrade().unwrap();
                let west_borrow = west.borrow();
                Some(Rc::downgrade(west_borrow.get_child(QuadPos::UpperRight).unwrap()))
            },
            QuadPos::LowerLeft => {
                let west = self.west.as_ref().unwrap().upgrade().unwrap();
                let west_borrow = west.borrow();
                Some(Rc::downgrade(west_borrow.get_child(QuadPos::LowerRight).unwrap()))
            },
            QuadPos::None => self.west.clone(),
        }
    }

    fn is_subdivided(&self) -> bool {
        self.children.is_some()
    }

    fn can_subdivide(&self) -> bool {
        let (indirect1, indirect2) = match self.pos {
            QuadPos::UpperLeft => (self.north.as_ref().unwrap(), self.west.as_ref().unwrap()),
            QuadPos::UpperRight => (self.north.as_ref().unwrap(), self.east.as_ref().unwrap()),
            QuadPos::LowerLeft => (self.south.as_ref().unwrap(), self.west.as_ref().unwrap()),
            QuadPos::LowerRight => (self.south.as_ref().unwrap(), self.east.as_ref().unwrap()),
            QuadPos::None => return true,
        };

        // TODO: write this in a more elegant way...
        let indirect1 = indirect1.upgrade().unwrap();
        let indirect2 = indirect2.upgrade().unwrap();
        let subdivided1 = indirect1.borrow().is_subdivided();
        let subdivided2 = indirect2.borrow().is_subdivided();
        subdivided1 && subdivided2
    }

    fn can_collapse(&self) -> bool {
        // TODO: It should be OK to use unwrap() here.
        let direct_north = self.direct_north();
        let direct_south = self.direct_south();
        let direct_east = self.direct_east();
        let direct_west = self.direct_west();

        if let Some(q) = direct_north {
            let q = q.upgrade().unwrap();
            let q_borrow = q.borrow();
            if q_borrow.is_subdivided() {
                let q1 = q_borrow.get_child(QuadPos::LowerLeft).unwrap();
                let q2 = q_borrow.get_child(QuadPos::LowerRight).unwrap();
                if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                    return false
                }
            }
        }

        if let Some(q) = direct_south {
            let q = q.upgrade().unwrap();
            let q_borrow = q.borrow();
            if q_borrow.is_subdivided() {
                let q1 = q_borrow.get_child(QuadPos::UpperLeft).unwrap();
                let q2 = q_borrow.get_child(QuadPos::UpperRight).unwrap();
                if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                    return false
                }
            }
        }

        if let Some(q) = direct_east {
            let q = q.upgrade().unwrap();
            let q_borrow = q.borrow();
            if q_borrow.is_subdivided() {
                let q1 = q_borrow.get_child(QuadPos::UpperLeft).unwrap();
                let q2 = q_borrow.get_child(QuadPos::LowerLeft).unwrap();
                if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                    return false
                }
            }
        }

        if let Some(q) = direct_west {
            let q = q.upgrade().unwrap();
            let q_borrow = q.borrow();
            if q_borrow.is_subdivided() {
                let q1 = q_borrow.get_child(QuadPos::UpperRight).unwrap();
                let q2 = q_borrow.get_child(QuadPos::LowerRight).unwrap();
                if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                    return false
                }
            }
        }

        true
    }

    fn subdivide(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        let half_quad_length = sphere.quad_length(self.cur_subdivision) / 2;

        let upper_left_obj = scene.create_object();
        let upper_left = scene.add_behaviour::<Quad>(&upper_left_obj).unwrap();
        let upper_right_obj = scene.create_object();
        let upper_right = scene.add_behaviour::<Quad>(&upper_right_obj).unwrap();
        let lower_left_obj = scene.create_object();
        let lower_left = scene.add_behaviour::<Quad>(&lower_left_obj).unwrap();
        let lower_right_obj = scene.create_object();
        let lower_right = scene.add_behaviour::<Quad>(&lower_right_obj).unwrap();

        let direct_north = self.direct_north();
        let direct_south = self.direct_south();
        let direct_east = self.direct_east();
        let direct_west = self.direct_west();

        {
            let mut upper_left = upper_left.borrow_mut();
            upper_left.plane = self.plane;
            upper_left.pos = QuadPos::UpperLeft;
            upper_left.north = direct_north.clone();
            upper_left.east = Some(Rc::downgrade(&upper_right));
            upper_left.south = Some(Rc::downgrade(&lower_left));
            upper_left.west = direct_west.clone();
            upper_left.cur_subdivision = self.cur_subdivision + 1;
            upper_left.base_coord = (self.base_coord.0, self.base_coord.1 + half_quad_length);
            upper_left.init(sphere, scene);
        }

        {
            let mut upper_right = upper_right.borrow_mut();
            upper_right.plane = self.plane;
            upper_right.pos = QuadPos::UpperRight;
            upper_right.north = direct_north.clone();
            upper_right.east = direct_east.clone();
            upper_right.south = Some(Rc::downgrade(&lower_right));
            upper_right.west = Some(Rc::downgrade(&upper_left));
            upper_right.cur_subdivision = self.cur_subdivision + 1;
            upper_right.base_coord = (self.base_coord.0 + half_quad_length, self.base_coord.1 + half_quad_length);
            upper_right.init(sphere, scene);
        }

        {
            let mut lower_left = lower_left.borrow_mut();
            lower_left.plane = self.plane;
            lower_left.pos = QuadPos::LowerLeft;
            lower_left.north = Some(Rc::downgrade(&upper_left));
            lower_left.east = Some(Rc::downgrade(&lower_right));
            lower_left.south = direct_south.clone();
            lower_left.west = direct_west.clone();
            lower_left.cur_subdivision = self.cur_subdivision + 1;
            lower_left.base_coord = (self.base_coord.0, self.base_coord.1);
            lower_left.init(sphere, scene);
        }

        {
            let mut lower_right = lower_right.borrow_mut();
            lower_right.plane = self.plane;
            lower_right.pos = QuadPos::LowerRight;
            lower_right.north = Some(Rc::downgrade(&upper_right));
            lower_right.east = direct_east.clone();
            lower_right.south = direct_south.clone();
            lower_right.west = Some(Rc::downgrade(&lower_left));
            lower_right.cur_subdivision = self.cur_subdivision + 1;
            lower_right.base_coord = (self.base_coord.0 + half_quad_length, self.base_coord.1);
            lower_right.init(sphere, scene);
        }

        self.children = Some([upper_left, upper_right, lower_left, lower_right]);
    }

    fn collapse(&mut self, scene: &mut Scene) {
        for q in self.children.as_ref().unwrap() {
            // TODO: reduce cloning
            let q_obj = q.borrow().behaviour().object(scene).unwrap().clone();
            scene.destroy_object(&q_obj);
        }
        self.children = None;
    }

    fn check_subdivision(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        if !self.is_subdivided()
            && self.cur_subdivision < sphere.max_subdivision
            && self.in_subdivision_range(sphere)
            && self.can_subdivide() {
            self.subdivide(sphere, scene);
        } else if self.is_subdivided() && self.in_collapse_range(sphere) && self.can_collapse() {
            self.collapse(scene);
        } else if self.is_subdivided() {
            for q in self.children.as_ref().unwrap() {
                q.borrow_mut().check_subdivision(sphere, scene);
            }
        }
    }

    #[allow(dead_code)]
    fn debug_find_quad(&self, path: &[QuadPos]) -> Rc<RefCell<Quad>> {
        if !self.is_subdivided() {
            panic!("Quad doesn't exist!");
        }

        let q = self.get_child(path[0]);
        if path.len() == 1 {
            return q.unwrap().clone()
        } else {
            q.unwrap().borrow().debug_find_quad(&path[1..])
        }
    }
}

impl BehaviourMessages for Quad {
    fn create(behaviour: Behaviour) -> Quad {
        Quad {
            behaviour: behaviour,
            plane: Plane::XP,
            pos: QuadPos::None,
            mesh: None,
            shader: None,
            material: None,
            base_coord: (0, 0),
            cur_subdivision: 0,
            children: None,
            north: None,
            south: None,
            east: None,
            west: None,
        }
    }

    fn start(&mut self, _scene: &mut Scene) {
    }

    fn update(&mut self, _scene: &mut Scene) {
    }

    fn destroy(&mut self, scene: &mut Scene) {
        self.mesh.take().map(|mesh| scene.destroy_mesh(&*mesh));
        self.material.take().map(|material| scene.destroy_material(&*material));
        self.shader.take().map(|shader| scene.destroy_shader(&*shader));
    }

    fn behaviour(&self) -> &Behaviour {
        &self.behaviour
    }

    fn mesh(&self) -> Option<&Mesh> {
        if !self.is_subdivided() {
            self.mesh.as_ref().map(|mesh| &**mesh)
        } else {
            None
        }
    }

    fn material(&self) -> Option<&Material> {
        if !self.is_subdivided() {
            self.material.as_ref().map(|material| &**material)
        } else {
            None
        }
    }
}

struct QuadSphere {
    behaviour: Behaviour,
    prev_instant: std::time::Instant,
    ninety_deg: Quaternion<f32>,
    quad_mesh_size: u16,
    max_subdivision: u32,
    max_coord: u32,
    centre_pos: Vector3<f32>,
    faces: Option<[Rc<RefCell<Quad>>; 6]>,
}

impl QuadSphere {
    fn init(&mut self, scene: &mut Scene, quad_mesh_size: u16, max_subdivision: u32) {
        assert!(quad_mesh_size > 1);
        let bits =  (quad_mesh_size as u32 - 1).leading_zeros();
        assert!(max_subdivision <= (bits - 1));

        self.quad_mesh_size = quad_mesh_size;
        self.max_subdivision = max_subdivision;
        self.max_coord = (1 << max_subdivision) * quad_mesh_size as u32;

        let xp_quad_obj = scene.create_object();
        let xp_quad = scene.add_behaviour::<Quad>(&xp_quad_obj).unwrap();
        let xn_quad_obj = scene.create_object();
        let xn_quad = scene.add_behaviour::<Quad>(&xn_quad_obj).unwrap();
        let yp_quad_obj = scene.create_object();
        let yp_quad = scene.add_behaviour::<Quad>(&yp_quad_obj).unwrap();
        let yn_quad_obj = scene.create_object();
        let yn_quad = scene.add_behaviour::<Quad>(&yn_quad_obj).unwrap();
        let zp_quad_obj = scene.create_object();
        let zp_quad = scene.add_behaviour::<Quad>(&zp_quad_obj).unwrap();
        let zn_quad_obj = scene.create_object();
        let zn_quad = scene.add_behaviour::<Quad>(&zn_quad_obj).unwrap();

        {
            let mut xp_quad = xp_quad.borrow_mut();
            xp_quad.plane = Plane::XP;
            xp_quad.pos = QuadPos::None;
            xp_quad.north = Some(Rc::downgrade(&yp_quad));
            xp_quad.south = Some(Rc::downgrade(&yn_quad));
            xp_quad.east = Some(Rc::downgrade(&zn_quad));
            xp_quad.west = Some(Rc::downgrade(&zp_quad));
            xp_quad.cur_subdivision = 0;
            xp_quad.base_coord = (0, 0);
            xp_quad.init(self, scene);
        }

        {
            let mut xn_quad = xn_quad.borrow_mut();
            xn_quad.plane = Plane::XN;
            xn_quad.pos = QuadPos::None;
            xn_quad.north = Some(Rc::downgrade(&yp_quad));
            xn_quad.south = Some(Rc::downgrade(&yn_quad));
            xn_quad.east = Some(Rc::downgrade(&zp_quad));
            xn_quad.west = Some(Rc::downgrade(&zn_quad));
            xn_quad.cur_subdivision = 0;
            xn_quad.base_coord = (0, 0);
            xn_quad.init(self, scene);
        }

        {
            let mut yp_quad = yp_quad.borrow_mut();
            yp_quad.plane = Plane::YP;
            yp_quad.pos = QuadPos::None;
            yp_quad.north = Some(Rc::downgrade(&zn_quad));
            yp_quad.south = Some(Rc::downgrade(&zp_quad));
            yp_quad.east = Some(Rc::downgrade(&xp_quad));
            yp_quad.west = Some(Rc::downgrade(&xn_quad));
            yp_quad.cur_subdivision = 0;
            yp_quad.base_coord = (0, 0);
            yp_quad.init(self, scene);
        }

        {
            let mut yn_quad = yn_quad.borrow_mut();
            yn_quad.plane = Plane::YN;
            yn_quad.pos = QuadPos::None;
            yn_quad.north = Some(Rc::downgrade(&zp_quad));
            yn_quad.south = Some(Rc::downgrade(&zn_quad));
            yn_quad.east = Some(Rc::downgrade(&xp_quad));
            yn_quad.west = Some(Rc::downgrade(&xn_quad));
            yn_quad.cur_subdivision = 0;
            yn_quad.base_coord = (0, 0);
            yn_quad.init(self, scene);
        }

        {
            let mut zp_quad = zp_quad.borrow_mut();
            zp_quad.plane = Plane::ZP;
            zp_quad.pos = QuadPos::None;
            zp_quad.north = Some(Rc::downgrade(&yp_quad));
            zp_quad.south = Some(Rc::downgrade(&yn_quad));
            zp_quad.east = Some(Rc::downgrade(&xp_quad));
            zp_quad.west = Some(Rc::downgrade(&xn_quad));
            zp_quad.cur_subdivision = 0;
            zp_quad.base_coord = (0, 0);
            zp_quad.init(self, scene);
        }

        {
            let mut zn_quad = zn_quad.borrow_mut();
            zn_quad.plane = Plane::ZN;
            zn_quad.pos = QuadPos::None;
            zn_quad.north = Some(Rc::downgrade(&yp_quad));
            zn_quad.south = Some(Rc::downgrade(&yn_quad));
            zn_quad.east = Some(Rc::downgrade(&xn_quad));
            zn_quad.west = Some(Rc::downgrade(&xp_quad));
            zn_quad.cur_subdivision = 0;
            zn_quad.base_coord = (0, 0);
            zn_quad.init(self, scene);
        }

        // TODO: reduce cloning
        let self_object = self.behaviour().object(&scene).unwrap().clone();
        scene.set_object_parent(&xp_quad_obj, Some(&self_object));
        scene.set_object_parent(&xn_quad_obj, Some(&self_object));
        scene.set_object_parent(&yp_quad_obj, Some(&self_object));
        scene.set_object_parent(&yn_quad_obj, Some(&self_object));
        scene.set_object_parent(&zp_quad_obj, Some(&self_object));
        scene.set_object_parent(&zn_quad_obj, Some(&self_object));

        self.faces = Some([xp_quad, xn_quad, yp_quad, yn_quad, zp_quad, zn_quad]);
    }

    fn quad_length(&self, level: u32) -> u32 {
        (1 << (self.max_subdivision - level)) * (self.quad_mesh_size as u32)
    }

    fn centre_pos(&self) -> Vector3<f32> {
        self.centre_pos
    }

    #[allow(dead_code)]
    fn debug_find_quad(&self, plane: Plane, path: &[QuadPos]) -> Rc<RefCell<Quad>> {
        let q = match plane {
            Plane::XP => &self.faces.as_ref().unwrap()[0],
            Plane::XN => &self.faces.as_ref().unwrap()[1],
            Plane::YP => &self.faces.as_ref().unwrap()[2],
            Plane::YN => &self.faces.as_ref().unwrap()[3],
            Plane::ZP => &self.faces.as_ref().unwrap()[4],
            Plane::ZN => &self.faces.as_ref().unwrap()[5],
        };

        if path.len() == 0 {
            return q.clone()
        } else {
            q.borrow().debug_find_quad(path)
        }
    }
}

impl BehaviourMessages for QuadSphere {
    fn create(behaviour: Behaviour) -> QuadSphere {
        QuadSphere {
            behaviour: behaviour,
            prev_instant: std::time::Instant::now(),
            ninety_deg: Quaternion::from(Euler { x: Deg(0.0), y: Deg(45.0), z: Deg(0.0) }),
            quad_mesh_size: 0,
            max_subdivision: 0,
            max_coord: 0,
            centre_pos: Vector3::unit_z(),
            faces: None,
        }
    }

    fn start(&mut self, _scene: &mut Scene) {
        self.prev_instant = std::time::Instant::now();
    }

    fn update(&mut self, scene: &mut Scene) {
        let diff = self.prev_instant.elapsed();
        self.prev_instant = std::time::Instant::now();
        let secs = diff.as_secs() as f32 + diff.subsec_nanos() as f32 / 1000000000.0;

        let dps = 20.0;
        let change_rot = Quaternion::one().nlerp(self.ninety_deg, (dps / 45.0) * secs);

        // TODO: reduce cloning
        let self_object = self.behaviour.object(scene).unwrap().clone();
        let rot = self_object.local_rot(scene).unwrap();
        let rot = rot * change_rot;
        self_object.set_local_rot(scene, rot).unwrap();

        self.centre_pos = rot.invert() * Vector3::unit_z();
        for i in 0..6 {
            let q = self.faces.as_ref().unwrap()[i].clone();
            q.borrow_mut().check_subdivision(self, scene);
        }
    }

    fn destroy(&mut self, _scene: &mut Scene) {
    }

    fn behaviour(&self) -> &Behaviour {
        &self.behaviour
    }

    fn mesh(&self) -> Option<&Mesh> {
        None
    }

    fn material(&self) -> Option<&Material> {
        None
    }
}

fn map_vertcoord(coord: VertCoord, max_coord: u32) -> Vector3<f32> {
    let (x, y, z) = match coord {
        VertCoord(Plane::XP, a, b) => (max_coord, b, max_coord - a),
        VertCoord(Plane::XN, a, b) => (0, b, a),
        VertCoord(Plane::YP, a, b) => (a, max_coord, max_coord - b),
        VertCoord(Plane::YN, a, b) => (a, 0, b),
        VertCoord(Plane::ZP, a, b) => (a, b, max_coord),
        VertCoord(Plane::ZN, a, b) => (max_coord - a, b, 0),
    };
    Vector3::new(-1.0 + x as f32 * 2.0 / max_coord as f32,
                 -1.0 + y as f32 * 2.0 / max_coord as f32,
                 -1.0 + z as f32 * 2.0 / max_coord as f32)
}



fn main() {
    let display = glium::glutin::WindowBuilder::new()
        .with_multisampling(8)
        .with_depth_buffer(24)
        .build_glium().unwrap();

    let mut scene = Scene::new(display);
    let mut _camera = scene.create_camera();
    //camera.near_clip = 0.1f32;
    //camera.far_clip = 1000f32;

    let quad_sphere_obj = scene.create_object();
    let quad_sphere = scene.add_behaviour::<QuadSphere>(&quad_sphere_obj).unwrap();
    quad_sphere.borrow_mut().init(&mut scene, 8, 5);
    quad_sphere_obj.set_world_pos(&mut scene, Vector3::new(0.0, 0.0, -2.5)).unwrap();
    //quad_sphere.borrow().object().set_world_rot(&mut scene, Quaternion::from(Euler { x: Deg(45.0), y: Deg(0.0), z: Deg(0.0) })).unwrap();

    loop {
        if !scene.do_frame() {
            break
        }
    }
}
