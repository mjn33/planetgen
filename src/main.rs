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

#![windows_subsystem="windows"]

#![feature(retain_hash_collection)]

#[macro_use]
extern crate bitflags;
extern crate cgmath;
extern crate sdl2;
extern crate noise;
extern crate num;
extern crate planetgen_engine;
extern crate png;

mod colour_curve;
mod common;
mod gen;
mod heightmap;
#[cfg(test)]
mod test;

use sdl2::keyboard::Scancode;

use std::cell::{Cell, RefCell, RefMut};
use std::collections::HashMap;
use std::fs::File;
use std::rc::{Rc, Weak};
use std::sync::mpsc::{channel, Sender, Receiver, TryRecvError};
use std::thread;

use cgmath::{Deg, Euler, InnerSpace, Quaternion, Rotation, Vector3};

use noise::module::{Module, Perlin, ScaleBias};
use noise::noisegen::NoiseQuality;

use num::{Zero, One};

use planetgen_engine::{BehaviourMessages, Camera, Handle, Material, Mesh, MeshRenderer, Object,
                       Scene, Shader, UniformValue};

use common::{map_quad_pos, map_quad_side, map_vec_pos, Plane, QuadPos, QuadSide, QuadSideFlags,
             QUAD_SIDE_FLAGS_NONE, QUAD_SIDE_FLAGS_NORTH, QUAD_SIDE_FLAGS_SOUTH,
             QUAD_SIDE_FLAGS_EAST, QUAD_SIDE_FLAGS_WEST, QUAD_SIDE_FLAGS_ALL};
use colour_curve::ColourCurve;
use gen::{gen_indices, vert_off};
use heightmap::Heightmap;

fn create_generator() -> Box<Module + Send> {
    const SEED: i32 = 600;
    const FREQUENCY: f64 = 20.0;
    const LACUNARITY: f64 = 2.208984375;

    let mut perlin = Perlin::default();
    perlin.set_seed(SEED);
    perlin.set_frequency(FREQUENCY);
    perlin.set_persistence(0.5);
    perlin.set_lacunarity(LACUNARITY);
    perlin.set_octave_count(14);
    perlin.set_quality(NoiseQuality::Standard);

    let mut sb = ScaleBias::new(perlin);
    sb.set_scale(0.5);
    sb.set_bias(0.5);

    Box::new(sb)
}

fn load_image(filename: &str) -> Result<(Box<[u8]>, u32, u32), String> {
    let file = try!(File::open(filename).map_err(|_| "Failed to open file"));
    let decoder = png::Decoder::new(file);
    let (info, mut reader) = try!(decoder
                                      .read_info()
                                      .map_err(|e| format!("Failed to create decoder: {}", e)));

    let mut img_data = vec![0; info.buffer_size()];
    try!(reader
             .next_frame(&mut img_data)
             .map_err(|e| format!("Failed to read image data: {}", e)));

    Ok((img_data.into_boxed_slice(), info.width, info.height))
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct VertCoord(Plane, i32, i32);

struct Quad {
    object: Handle<Object>,
    plane: Plane,
    pos: QuadPos,
    mrenderer: Option<Handle<MeshRenderer>>,
    mesh: Option<Handle<Mesh>>,
    base_coord: (i32, i32),
    cur_subdivision: i32,
    mid_coord_pos: Vector3<f64>,
    /// A tuple containing the sine and cosine of an angle which encloses the
    /// entire quad.
    angle_size: (f64, f64),
    patch_flags: QuadSideFlags,

    /// A copy of the normals of this quad before merging.
    unmerged_normals: Vec<Vector3<f32>>,

    /// True if this quad's patching has changed.
    patching_dirty: bool,
    /// Describes which sides of this quad need to have quad normals be merged
    /// with neighbouring quads.
    needs_normal_merge: QuadSideFlags,
    /// True if this quad is currently being rendered.
    render: bool,

    /// Pointer to this quad
    self_ptr: Option<Weak<RefCell<Quad>>>,
    /// Contains the child quads of this quad in the following order: upper
    /// left, upper right, lower left and lower right. Is `None` if this quad
    /// isn't subdivided.
    children: Option<[Rc<RefCell<Quad>>; 4]>,
    north: Option<Weak<RefCell<Quad>>>,
    south: Option<Weak<RefCell<Quad>>>,
    east: Option<Weak<RefCell<Quad>>>,
    west: Option<Weak<RefCell<Quad>>>,
}

impl Quad {
    fn new(scene: &mut Scene) -> Quad {
        let object = scene.create_object();
        Quad {
            object: object,
            plane: Plane::XP,
            pos: QuadPos::None,
            mrenderer: None,
            mesh: None,
            base_coord: (0, 0),
            cur_subdivision: 0,
            mid_coord_pos: Vector3::new(0.0, 0.0, 0.0),
            angle_size: (1.0, 1.0),
            patch_flags: QUAD_SIDE_FLAGS_NONE,

            unmerged_normals: Vec::new(),

            patching_dirty: false,
            needs_normal_merge: QUAD_SIDE_FLAGS_NONE,
            render: false,

            self_ptr: None,
            children: None,
            north: None,
            south: None,
            east: None,
            west: None,
        }
    }

    /// Calculates the coordinates of the middle of this quad.
    fn mid_coord_pos(&self, sphere: &QuadSphere) -> Vector3<f64> {
        let half_quad_length = sphere.quad_length(self.cur_subdivision) / 2;
        let mid_coord = (self.base_coord.0 + half_quad_length,
                         self.base_coord.1 + half_quad_length);
        sphere.vc_to_pos(VertCoord(self.plane, mid_coord.0, mid_coord.1)).normalize()
    }

    fn angle_size(&self, sphere: &QuadSphere) -> (f64, f64) {
        let quad_length = sphere.quad_length(self.cur_subdivision);
        let c0 = (self.base_coord.0, self.base_coord.1);
        let c1 = (self.base_coord.0 + quad_length, self.base_coord.1);
        let c2 = (self.base_coord.0, self.base_coord.1 + quad_length);
        let c3 = (self.base_coord.0 + quad_length, self.base_coord.1 + quad_length);

        let c0_pos = sphere.vc_to_pos(VertCoord(self.plane, c0.0, c0.1)).normalize();
        let c1_pos = sphere.vc_to_pos(VertCoord(self.plane, c1.0, c1.1)).normalize();
        let c2_pos = sphere.vc_to_pos(VertCoord(self.plane, c2.0, c2.1)).normalize();
        let c3_pos = sphere.vc_to_pos(VertCoord(self.plane, c3.0, c3.1)).normalize();

        let c0_cos_alpha = Quad::dot(self.mid_coord_pos, c0_pos);
        let c1_cos_alpha = Quad::dot(self.mid_coord_pos, c1_pos);
        let c2_cos_alpha = Quad::dot(self.mid_coord_pos, c2_pos);
        let c3_cos_alpha = Quad::dot(self.mid_coord_pos, c3_pos);

        // Use the furthest away corner (lowest cos(alpha))
        let tmp1 = f64::min(c0_cos_alpha, c1_cos_alpha);
        let tmp2 = f64::min(c2_cos_alpha, c3_cos_alpha);
        let cos_alpha = f64::min(tmp1, tmp2);

        let sin_alpha = f64::sqrt(1.0 - cos_alpha * cos_alpha);

        (sin_alpha, cos_alpha)
    }

    /// Computes the dot product of two vectors, assuming both vectors are unit
    /// vectors.
    fn dot(a: Vector3<f64>, b: Vector3<f64>) -> f64 {
        let c = a - b;
        let c_mag2 = c.magnitude2();
        let cos_gamma = (2.0 - c_mag2) * 0.5;
        cos_gamma
    }

    /// Returns `true` if this quad is within subdivision range.
    fn in_subdivision_range(&self, sphere: &QuadSphere) -> bool {
        if self.cur_subdivision == sphere.max_subdivision() {
            return false;
        }

        let range = sphere.subdivide_range(self.cur_subdivision);

        let centre_pos = sphere.centre_pos();
        let cur_range = Quad::dot(centre_pos, self.mid_coord_pos);

        // Note: comparison may seem swapped since lower values mean a greater
        // angle / arc.
        cur_range >= range
    }

    /// Returns `true` if this quad is within collapse range.
    fn in_collapse_range(&self, sphere: &QuadSphere) -> bool {
        let range = sphere.collapse_range(self.cur_subdivision);

        let centre_pos = sphere.centre_pos();
        let cur_range = Quad::dot(centre_pos, self.mid_coord_pos);

        // Note: comparison may seem swapped since higher values mean a smaller
        // angle / arc.
        cur_range <= range
    }

    /// Returns the child quad at the given position.
    ///
    /// # Panics
    ///
    /// Panics if the quad isn't subdivided.
    fn get_child(&self, pos: QuadPos) -> &Rc<RefCell<Quad>> {
        self.get_child_opt(pos).expect("Expected quad to be subdivided")
    }

    /// Returns the child quad at the given position, or None if the quad isn't
    /// subdivided.
    fn get_child_opt(&self, pos: QuadPos) -> Option<&Rc<RefCell<Quad>>> {
        self.children.as_ref().map(|children| &children[pos.to_idx()])
    }

    fn north(&self) -> Rc<RefCell<Quad>> {
        self.north.as_ref().unwrap().upgrade().unwrap()
    }

    fn south(&self) -> Rc<RefCell<Quad>> {
        self.south.as_ref().unwrap().upgrade().unwrap()
    }

    fn east(&self) -> Rc<RefCell<Quad>> {
        self.east.as_ref().unwrap().upgrade().unwrap()
    }

    fn west(&self) -> Rc<RefCell<Quad>> {
        self.west.as_ref().unwrap().upgrade().unwrap()
    }

    fn direct_north(&self) -> Option<Rc<RefCell<Quad>>> {
        match self.pos {
            QuadPos::SouthWest | QuadPos::SouthEast | QuadPos::None => Some(self.north()),
            QuadPos::NorthWest => {
                let north = self.north();
                let north_borrow = north.borrow();
                let pos = map_quad_pos(QuadPos::SouthWest, self.plane, north_borrow.plane);
                north_borrow.get_child_opt(pos).map(Rc::clone)
            }
            QuadPos::NorthEast => {
                let north = self.north();
                let north_borrow = north.borrow();
                let pos = map_quad_pos(QuadPos::SouthEast, self.plane, north_borrow.plane);
                north_borrow.get_child_opt(pos).map(Rc::clone)
            }
        }
    }

    fn direct_south(&self) -> Option<Rc<RefCell<Quad>>> {
        match self.pos {
            QuadPos::NorthWest | QuadPos::NorthEast | QuadPos::None => Some(self.south()),
            QuadPos::SouthWest => {
                let south = self.south();
                let south_borrow = south.borrow();
                let pos = map_quad_pos(QuadPos::NorthWest, self.plane, south_borrow.plane);
                south_borrow.get_child_opt(pos).map(Rc::clone)
            }
            QuadPos::SouthEast => {
                let south = self.south();
                let south_borrow = south.borrow();
                let pos = map_quad_pos(QuadPos::NorthEast, self.plane, south_borrow.plane);
                south_borrow.get_child_opt(pos).map(Rc::clone)
            }
        }
    }

    fn direct_east(&self) -> Option<Rc<RefCell<Quad>>> {
        match self.pos {
            QuadPos::NorthWest | QuadPos::SouthWest | QuadPos::None => Some(self.east()),
            QuadPos::NorthEast => {
                let east = self.east();
                let east_borrow = east.borrow();
                let pos = map_quad_pos(QuadPos::NorthWest, self.plane, east_borrow.plane);
                east_borrow.get_child_opt(pos).map(Rc::clone)
            }
            QuadPos::SouthEast => {
                let east = self.east();
                let east_borrow = east.borrow();
                let pos = map_quad_pos(QuadPos::SouthWest, self.plane, east_borrow.plane);
                east_borrow.get_child_opt(pos).map(Rc::clone)
            }
        }
    }

    fn direct_west(&self) -> Option<Rc<RefCell<Quad>>> {
        match self.pos {
            QuadPos::NorthEast | QuadPos::SouthEast | QuadPos::None => Some(self.west()),
            QuadPos::NorthWest => {
                let west = self.west();
                let west_borrow = west.borrow();
                let pos = map_quad_pos(QuadPos::NorthEast, self.plane, west_borrow.plane);
                west_borrow.get_child_opt(pos).map(Rc::clone)
            }
            QuadPos::SouthWest => {
                let west = self.west();
                let west_borrow = west.borrow();
                let pos = map_quad_pos(QuadPos::SouthEast, self.plane, west_borrow.plane);
                west_borrow.get_child_opt(pos).map(Rc::clone)
            }
        }
    }

    /// Returns `true` if this quad is subdivided.
    fn is_subdivided(&self) -> bool {
        self.children.is_some()
    }

    /// Returns `true` if this quad can be subdivided without violating the
    /// constraint that neighbouring quads can only be, at a maximum, one
    /// subdivision level higher or lower.
    fn can_subdivide(&self) -> bool {
        let (indirect1, indirect2) = match self.pos {
            QuadPos::NorthWest => (self.north(), self.west()),
            QuadPos::NorthEast => (self.north(), self.east()),
            QuadPos::SouthWest => (self.south(), self.west()),
            QuadPos::SouthEast => (self.south(), self.east()),
            QuadPos::None => return true,
        };

        let subdivided = indirect1.borrow().is_subdivided() && indirect2.borrow().is_subdivided();
        subdivided
    }

    /// Returns `true` if this quad can be collapsed without violating the
    /// constraint that neighbouring quads can only be, at a maximum, one
    /// subdivision level higher or lower.
    fn can_collapse(&self) -> bool {
        assert!(self.is_subdivided(),
                "can_collapse() should only be called on subdivided quads");
        let direct_north = self.direct_north().unwrap();
        let direct_south = self.direct_south().unwrap();
        let direct_east = self.direct_east().unwrap();
        let direct_west = self.direct_west().unwrap();

        let direct_north = direct_north.borrow();
        if direct_north.is_subdivided() {
            let pos1 = map_quad_pos(QuadPos::SouthWest, self.plane, direct_north.plane);
            let pos2 = map_quad_pos(QuadPos::SouthEast, self.plane, direct_north.plane);
            let q1 = direct_north.get_child(pos1);
            let q2 = direct_north.get_child(pos2);
            if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                return false;
            }
        }

        let direct_south = direct_south.borrow();
        if direct_south.is_subdivided() {
            let pos1 = map_quad_pos(QuadPos::NorthWest, self.plane, direct_south.plane);
            let pos2 = map_quad_pos(QuadPos::NorthEast, self.plane, direct_south.plane);
            let q1 = direct_south.get_child(pos1);
            let q2 = direct_south.get_child(pos2);
            if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                return false;
            }
        }

        let direct_east = direct_east.borrow();
        if direct_east.is_subdivided() {
            let pos1 = map_quad_pos(QuadPos::NorthWest, self.plane, direct_east.plane);
            let pos2 = map_quad_pos(QuadPos::SouthWest, self.plane, direct_east.plane);
            let q1 = direct_east.get_child(pos1);
            let q2 = direct_east.get_child(pos2);
            if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                return false;
            }
        }

        let direct_west = direct_west.borrow();
        if direct_west.is_subdivided() {
            let pos1 = map_quad_pos(QuadPos::NorthEast, self.plane, direct_west.plane);
            let pos2 = map_quad_pos(QuadPos::SouthEast, self.plane, direct_west.plane);
            let q1 = direct_west.get_child(pos1);
            let q2 = direct_west.get_child(pos2);
            if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                return false;
            }
        }

        for q in self.children.as_ref().unwrap() {
            if q.borrow().is_subdivided() {
                return false;
            }
        }

        true
    }

    fn subdivide(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        let half_quad_length = sphere.quad_length(self.cur_subdivision) / 2;

        let upper_left_base_coord = (self.base_coord.0, self.base_coord.1 + half_quad_length);
        let upper_right_base_coord = (self.base_coord.0 + half_quad_length,
                                      self.base_coord.1 + half_quad_length);
        let lower_left_base_coord = (self.base_coord.0, self.base_coord.1);
        let lower_right_base_coord = (self.base_coord.0 + half_quad_length, self.base_coord.1);

        let task_id = TaskId(VertCoord(self.plane, self.base_coord.0, self.base_coord.1),
                             self.cur_subdivision);
        let result = sphere.task_manager().get_task(task_id);

        let result = match result {
            TaskStatus::Complete(result) => result,
            TaskStatus::NotComplete => {
                sphere.task_manager().reset_task_lifetime(task_id);
                return;
            }
            TaskStatus::NotStarted => {
                sphere.task_manager().add_task(task_id, TaskData(()));
                return;
            }
        };

        let sphere_obj = sphere.object;
        let upper_left = sphere.quad_pool().get_quad(scene);
        let upper_right = sphere.quad_pool().get_quad(scene);
        let lower_left = sphere.quad_pool().get_quad(scene);
        let lower_right = sphere.quad_pool().get_quad(scene);

        let direct_north = self.direct_north();
        let direct_south = self.direct_south();
        let direct_east = self.direct_east();
        let direct_west = self.direct_west();

        {
            let self_ptr = Rc::downgrade(&upper_left);
            let mut upper_left = upper_left.borrow_mut();
            upper_left.plane = self.plane;
            upper_left.pos = QuadPos::NorthWest;
            upper_left.unmerged_normals.copy_from_slice(&result.q1.vnorm);
            upper_left.patching_dirty = true;
            upper_left.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
            upper_left.self_ptr = Some(self_ptr);
            upper_left.north = direct_north.as_ref().map(Rc::downgrade);
            upper_left.east = Some(Rc::downgrade(&upper_right));
            upper_left.south = Some(Rc::downgrade(&lower_left));
            upper_left.west = direct_west.as_ref().map(Rc::downgrade);
            upper_left.cur_subdivision = self.cur_subdivision + 1;
            upper_left.base_coord = upper_left_base_coord;
            upper_left.mid_coord_pos = upper_left.mid_coord_pos(sphere);
            upper_left.angle_size = upper_left.angle_size(sphere);

            scene.set_object_parent(upper_left.object, Some(sphere_obj));
            let mesh = upper_left.mesh.as_ref().unwrap();
            *mesh.vpos_mut(scene).unwrap() = result.q1.vpos;
            *mesh.vcolour_mut(scene).unwrap() = result.q1.vcolour;
            *mesh.vnorm_mut(scene).unwrap() = result.q1.vnorm;
        }
        sphere.queue_normal_update(upper_left.clone());

        {
            let self_ptr = Rc::downgrade(&upper_right);
            let mut upper_right = upper_right.borrow_mut();
            upper_right.plane = self.plane;
            upper_right.pos = QuadPos::NorthEast;
            upper_right.unmerged_normals.copy_from_slice(&result.q2.vnorm);
            upper_right.patching_dirty = true;
            upper_right.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
            upper_right.self_ptr = Some(self_ptr);
            upper_right.north = direct_north.as_ref().map(Rc::downgrade);
            upper_right.east = direct_east.as_ref().map(Rc::downgrade);
            upper_right.south = Some(Rc::downgrade(&lower_right));
            upper_right.west = Some(Rc::downgrade(&upper_left));
            upper_right.cur_subdivision = self.cur_subdivision + 1;
            upper_right.base_coord = upper_right_base_coord;
            upper_right.mid_coord_pos = upper_right.mid_coord_pos(sphere);
            upper_right.angle_size = upper_right.angle_size(sphere);

            scene.set_object_parent(upper_right.object, Some(sphere_obj));
            let mesh = upper_right.mesh.as_ref().unwrap();
            *mesh.vpos_mut(scene).unwrap() = result.q2.vpos;
            *mesh.vcolour_mut(scene).unwrap() = result.q2.vcolour;
            *mesh.vnorm_mut(scene).unwrap() = result.q2.vnorm;
        }
        sphere.queue_normal_update(upper_right.clone());

        {
            let self_ptr = Rc::downgrade(&lower_left);
            let mut lower_left = lower_left.borrow_mut();
            lower_left.plane = self.plane;
            lower_left.pos = QuadPos::SouthWest;
            lower_left.unmerged_normals.copy_from_slice(&result.q3.vnorm);
            lower_left.patching_dirty = true;
            lower_left.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
            lower_left.self_ptr = Some(self_ptr);
            lower_left.north = Some(Rc::downgrade(&upper_left));
            lower_left.east = Some(Rc::downgrade(&lower_right));
            lower_left.south = direct_south.as_ref().map(Rc::downgrade);
            lower_left.west = direct_west.as_ref().map(Rc::downgrade);
            lower_left.cur_subdivision = self.cur_subdivision + 1;
            lower_left.base_coord = lower_left_base_coord;
            lower_left.mid_coord_pos = lower_left.mid_coord_pos(sphere);
            lower_left.angle_size = lower_left.angle_size(sphere);

            scene.set_object_parent(lower_left.object, Some(sphere_obj));
            let mesh = lower_left.mesh.as_ref().unwrap();
            *mesh.vpos_mut(scene).unwrap() = result.q3.vpos;
            *mesh.vcolour_mut(scene).unwrap() = result.q3.vcolour;
            *mesh.vnorm_mut(scene).unwrap() = result.q3.vnorm;
        }
        sphere.queue_normal_update(lower_left.clone());

        {
            let self_ptr = Rc::downgrade(&lower_right);
            let mut lower_right = lower_right.borrow_mut();
            lower_right.plane = self.plane;
            lower_right.pos = QuadPos::SouthEast;
            lower_right.unmerged_normals.copy_from_slice(&result.q4.vnorm);
            lower_right.patching_dirty = true;
            lower_right.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
            lower_right.self_ptr = Some(self_ptr);
            lower_right.north = Some(Rc::downgrade(&upper_right));
            lower_right.east = direct_east.as_ref().map(Rc::downgrade);
            lower_right.south = direct_south.as_ref().map(Rc::downgrade);
            lower_right.west = Some(Rc::downgrade(&lower_left));
            lower_right.cur_subdivision = self.cur_subdivision + 1;
            lower_right.base_coord = lower_right_base_coord;
            lower_right.mid_coord_pos = lower_right.mid_coord_pos(sphere);
            lower_right.angle_size = lower_right.angle_size(sphere);

            scene.set_object_parent(lower_right.object, Some(sphere_obj));
            let mesh = lower_right.mesh.as_ref().unwrap();
            *mesh.vpos_mut(scene).unwrap() = result.q4.vpos;
            *mesh.vcolour_mut(scene).unwrap() = result.q4.vcolour;
            *mesh.vnorm_mut(scene).unwrap() = result.q4.vnorm;
        }
        sphere.queue_normal_update(lower_right.clone());

        let direct_north = direct_north.unwrap();
        let direct_south = direct_south.unwrap();
        let direct_east = direct_east.unwrap();
        let direct_west = direct_west.unwrap();

        let north_subdivided = direct_north.borrow().is_subdivided();
        let south_subdivided = direct_south.borrow().is_subdivided();
        let east_subdivided = direct_east.borrow().is_subdivided();
        let west_subdivided = direct_west.borrow().is_subdivided();

        if north_subdivided {
            let north_borrow = direct_north.borrow();
            let pos1 = map_quad_pos(QuadPos::SouthWest, self.plane, north_borrow.plane);
            let pos2 = map_quad_pos(QuadPos::SouthEast, self.plane, north_borrow.plane);
            let q1 = north_borrow.get_child(pos1);
            let q2 = north_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::South, self.plane, north_borrow.plane));
            q1_borrow.patch_flags &= !flags;
            q1_borrow.patching_dirty = true;
            q1_borrow.needs_normal_merge |= flags;
            q2_borrow.patch_flags &= !flags;
            q2_borrow.patching_dirty = true;
            q2_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut north_borrow = direct_north.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::South, self.plane, north_borrow.plane));
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            north_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(direct_north.clone());
        }

        if south_subdivided {
            let south_borrow = direct_south.borrow();
            let pos1 = map_quad_pos(QuadPos::NorthWest, self.plane, south_borrow.plane);
            let pos2 = map_quad_pos(QuadPos::NorthEast, self.plane, south_borrow.plane);
            let q1 = south_borrow.get_child(pos1);
            let q2 = south_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::North, self.plane, south_borrow.plane));
            q1_borrow.patch_flags &= !flags;
            q1_borrow.patching_dirty = true;
            q1_borrow.needs_normal_merge |= flags;
            q2_borrow.patch_flags &= !flags;
            q2_borrow.patching_dirty = true;
            q2_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut south_borrow = direct_south.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::North, self.plane, south_borrow.plane));
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            south_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(direct_south.clone());
        }

        if east_subdivided {
            let east_borrow = direct_east.borrow();
            let pos1 = map_quad_pos(QuadPos::NorthWest, self.plane, east_borrow.plane);
            let pos2 = map_quad_pos(QuadPos::SouthWest, self.plane, east_borrow.plane);
            let q1 = east_borrow.get_child(pos1);
            let q2 = east_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::West, self.plane, east_borrow.plane));
            q1_borrow.patch_flags &= !flags;
            q1_borrow.patching_dirty = true;
            q1_borrow.needs_normal_merge |= flags;
            q2_borrow.patch_flags &= !flags;
            q2_borrow.patching_dirty = true;
            q2_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut east_borrow = direct_east.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::West, self.plane, east_borrow.plane));
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            east_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(direct_east.clone());
        }

        if west_subdivided {
            let west_borrow = direct_west.borrow();
            let pos1 = map_quad_pos(QuadPos::NorthEast, self.plane, west_borrow.plane);
            let pos2 = map_quad_pos(QuadPos::SouthEast, self.plane, west_borrow.plane);
            let q1 = west_borrow.get_child(pos1);
            let q2 = west_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::East, self.plane, west_borrow.plane));
            q1_borrow.patch_flags &= !flags;
            q1_borrow.patching_dirty = true;
            q1_borrow.needs_normal_merge |= flags;
            q2_borrow.patch_flags &= !flags;
            q2_borrow.patching_dirty = true;
            q2_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut west_borrow = direct_west.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::East, self.plane, west_borrow.plane));
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            west_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(direct_west.clone());
        }

        {
            let mut upper_left = upper_left.borrow_mut();
            let mut upper_right = upper_right.borrow_mut();
            let mut lower_left = lower_left.borrow_mut();
            let mut lower_right = lower_right.borrow_mut();

            if !north_subdivided {
                upper_left.patch_flags |= QUAD_SIDE_FLAGS_NORTH;
                upper_right.patch_flags |= QUAD_SIDE_FLAGS_NORTH;
            }

            if !south_subdivided {
                lower_left.patch_flags |= QUAD_SIDE_FLAGS_SOUTH;
                lower_right.patch_flags |= QUAD_SIDE_FLAGS_SOUTH;
            }

            if !east_subdivided {
                upper_right.patch_flags |= QUAD_SIDE_FLAGS_EAST;
                lower_right.patch_flags |= QUAD_SIDE_FLAGS_EAST;
            }

            if !west_subdivided {
                upper_left.patch_flags |= QUAD_SIDE_FLAGS_WEST;
                lower_left.patch_flags |= QUAD_SIDE_FLAGS_WEST;
            }
        }

        // `self` visibility updated in `check_subdivision`
        upper_left.borrow_mut().update_visibility(sphere, scene);
        upper_right.borrow_mut().update_visibility(sphere, scene);
        lower_left.borrow_mut().update_visibility(sphere, scene);
        lower_right.borrow_mut().update_visibility(sphere, scene);

        self.patching_dirty = false;
        self.needs_normal_merge = QUAD_SIDE_FLAGS_NONE;

        self.children = Some([upper_left, upper_right, lower_left, lower_right]);
    }

    fn collapse(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        for q in self.children.as_ref().unwrap() {
            sphere.quad_pool().recycle_quad(scene, q.clone());
            let mut q_borrow = q.borrow_mut();
            q_borrow.patching_dirty = false;
            q_borrow.needs_normal_merge = QUAD_SIDE_FLAGS_NONE;
        }

        self.children = None;
        // `self` visibility updated in `check_subdivision`

        let direct_north = self.direct_north().unwrap();
        let direct_south = self.direct_south().unwrap();
        let direct_east = self.direct_east().unwrap();
        let direct_west = self.direct_west().unwrap();

        let north_subdivided = direct_north.borrow().is_subdivided();
        let south_subdivided = direct_south.borrow().is_subdivided();
        let east_subdivided = direct_east.borrow().is_subdivided();
        let west_subdivided = direct_west.borrow().is_subdivided();

        if north_subdivided {
            let north_borrow = direct_north.borrow();
            let pos1 = map_quad_pos(QuadPos::SouthWest, self.plane, north_borrow.plane);
            let pos2 = map_quad_pos(QuadPos::SouthEast, self.plane, north_borrow.plane);
            let q1 = north_borrow.get_child(pos1);
            let q2 = north_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::South, self.plane, north_borrow.plane));
            q1_borrow.patch_flags |= flags;
            q1_borrow.patching_dirty = true;
            q1_borrow.needs_normal_merge |= flags;
            q2_borrow.patch_flags |= flags;
            q2_borrow.patching_dirty = true;
            q2_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut north_borrow = direct_north.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::South, self.plane, north_borrow.plane));
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            north_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(direct_north.clone());
        }

        if south_subdivided {
            let south_borrow = direct_south.borrow();
            let pos1 = map_quad_pos(QuadPos::NorthWest, self.plane, south_borrow.plane);
            let pos2 = map_quad_pos(QuadPos::NorthEast, self.plane, south_borrow.plane);
            let q1 = south_borrow.get_child(pos1);
            let q2 = south_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::North, self.plane, south_borrow.plane));
            q1_borrow.patch_flags |= flags;
            q1_borrow.patching_dirty = true;
            q1_borrow.needs_normal_merge |= flags;
            q2_borrow.patch_flags |= flags;
            q2_borrow.patching_dirty = true;
            q2_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut south_borrow = direct_south.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::North, self.plane, south_borrow.plane));
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            south_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(direct_south.clone());
        }

        if east_subdivided {
            let east_borrow = direct_east.borrow();
            let pos1 = map_quad_pos(QuadPos::NorthWest, self.plane, east_borrow.plane);
            let pos2 = map_quad_pos(QuadPos::SouthWest, self.plane, east_borrow.plane);
            let q1 = east_borrow.get_child(pos1);
            let q2 = east_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::West, self.plane, east_borrow.plane));
            q1_borrow.patch_flags |= flags;
            q1_borrow.patching_dirty = true;
            q1_borrow.needs_normal_merge |= flags;
            q2_borrow.patch_flags |= flags;
            q2_borrow.patching_dirty = true;
            q2_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut east_borrow = direct_east.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::West, self.plane, east_borrow.plane));
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            east_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(direct_east.clone());
        }

        if west_subdivided {
            let west_borrow = direct_west.borrow();
            let pos1 = map_quad_pos(QuadPos::NorthEast, self.plane, west_borrow.plane);
            let pos2 = map_quad_pos(QuadPos::SouthEast, self.plane, west_borrow.plane);
            let q1 = west_borrow.get_child(pos1);
            let q2 = west_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::East, self.plane, west_borrow.plane));
            q1_borrow.patch_flags |= flags;
            q1_borrow.patching_dirty = true;
            q1_borrow.needs_normal_merge |= flags;
            q2_borrow.patch_flags |= flags;
            q2_borrow.patching_dirty = true;
            q2_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut west_borrow = direct_west.borrow_mut();
            let flags =
                QuadSideFlags::from(map_quad_side(QuadSide::East, self.plane, west_borrow.plane));
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            west_borrow.needs_normal_merge |= flags;
            sphere.queue_normal_update(direct_west.clone());
        }

        self.patch_flags = QUAD_SIDE_FLAGS_NONE;
        self.patching_dirty = true;
        self.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
        sphere.queue_normal_update(self.self_ptr.as_ref().unwrap().upgrade().unwrap());
    }

    fn in_horizon_cull_range(&self, sphere: &QuadSphere) -> bool {
        let range = sphere.horizon_cull_range(self.angle_size);

        let centre_pos = sphere.centre_pos();
        let cur_range = Quad::dot(centre_pos, self.mid_coord_pos);

        // Note: comparison may seem swapped since higher values mean a smaller
        // angle / arc.
        cur_range <= range
    }

    fn update_visibility(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        let enabled = !self.is_subdivided() && !self.in_horizon_cull_range(sphere);
        if enabled != self.render {
            self.mrenderer.as_ref().unwrap().set_enabled(scene, enabled).unwrap();
            self.render = enabled;
        }
    }

    fn check_subdivision(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        if !self.is_subdivided()
            && self.cur_subdivision < sphere.max_subdivision()
            && self.in_subdivision_range(sphere)
            && self.can_subdivide() {
            self.subdivide(sphere, scene);
        } else if self.is_subdivided() && self.in_collapse_range(sphere) && self.can_collapse() {
            self.collapse(sphere, scene);
        } else if self.is_subdivided() {
            for q in self.children.as_ref().unwrap() {
                q.borrow_mut().check_subdivision(sphere, scene);
            }
        }

        self.update_visibility(sphere, scene);
    }

    /// Recurses through the quad tree destroying all quad objects.
    fn cleanup(&mut self, scene: &mut Scene) {
        if self.is_subdivided() {
            for q in self.children.as_ref().unwrap() {
                q.borrow_mut().cleanup(scene);
            }
        }

        scene.destroy_object(self.object);
    }

    fn update_normals(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        {
            let mesh = self.mesh.as_ref().unwrap();
            let indices = mesh.indices_mut(scene).unwrap();
            let new_indices = sphere.quad_pool().get_indices(self.patch_flags);
            indices.clear();
            indices.extend(new_indices);
        }
    }

    fn update_edge_normals(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        if self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_NORTH) {
            self.merge_side_normals(sphere, scene, QuadSide::North);
        }
        if self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_SOUTH) {
            self.merge_side_normals(sphere, scene, QuadSide::South);
        }
        if self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_EAST) {
            self.merge_side_normals(sphere, scene, QuadSide::East);
        }
        if self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_WEST) {
            self.merge_side_normals(sphere, scene, QuadSide::West);
        }

        if self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_NORTH) ||
           self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_WEST) {
            self.merge_corner_normal(sphere, scene, QuadPos::NorthWest);
        }
        if self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_NORTH) ||
           self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_EAST) {
            self.merge_corner_normal(sphere, scene, QuadPos::NorthEast);
        }
        if self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_SOUTH) ||
           self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_WEST) {
            self.merge_corner_normal(sphere, scene, QuadPos::SouthWest);
        }
        if self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_SOUTH) ||
           self.needs_normal_merge.contains(QUAD_SIDE_FLAGS_EAST) {
            self.merge_corner_normal(sphere, scene, QuadPos::SouthEast);
        }
    }

    #[allow(dead_code)]
    fn debug_get_stats(&self,
                       sphere: &QuadSphere,
                       level_count: &mut [usize],
                       visible_level_count: &mut [usize],
                       horizon_cull_count: &mut usize,
                       horizon_cull_visible_count: &mut usize) {
        let ihcr = self.in_horizon_cull_range(sphere);
        let subdivided = self.is_subdivided();

        if !subdivided && ihcr {
            *horizon_cull_visible_count += 1;
        }
        if ihcr {
            *horizon_cull_count += 1;
        }

        level_count[self.cur_subdivision as usize] += 1;
        if self.is_subdivided() {
            for q in self.children.as_ref().unwrap() {
                q.borrow().debug_get_stats(
                    sphere,
                    level_count,
                    visible_level_count,
                    horizon_cull_count,
                    horizon_cull_visible_count);
            }
        } else {
            visible_level_count[self.cur_subdivision as usize] += 1;
        }
    }

    #[allow(dead_code)]
    fn debug_find_quad(&self, path: &[QuadPos]) -> Rc<RefCell<Quad>> {
        if !self.is_subdivided() {
            panic!("Quad doesn't exist!");
        }

        let q = self.get_child(path[0]);
        if path.len() == 1 {
            q.clone()
        } else {
            q.borrow().debug_find_quad(&path[1..])
        }
    }
}

struct QuadGenResult {
    vpos: Vec<Vector3<f32>>,
    vcolour: Vec<Vector3<f32>>,
    vnorm: Vec<Vector3<f32>>,
}

/// Structure containing the set of quad-sphere parameters needed for quad mesh
/// generation. Currently two copies of this structure exist: one inside the
/// `QuadSphere` structure itself, and one in `TaskExecutor` for executing quad
/// subdivision tasks asynchronously.
struct QuadSphereParams {
    /// The dimensions of the mesh of a quad, the quad with have
    /// `(quad_mesh_size + 1) * (quad_mesh_size + 1)` vertices.
    quad_mesh_size: i32,
    /// The maximum subdivision level we are allowed to go to, e.g. a value of
    /// 1 would only allow us to subdivide the root quads once.
    max_subdivision: i32,
    /// The maximum valid x or y value for a `VertCoord`.
    max_coord: i32,
    /// The base radius for the quad-sphere
    radius: f64,
    /// The minimum allowed height for the quad-sphere, the minimum radius is
    /// thus `radius - min_height`.
    min_height: f64,
    /// The maximum allowed height for the quad-sphere, the maximum radius is
    /// thus `radius + max_height`.
    max_height: f64,
    generator: Box<Module + Send>,
    heightmap: Heightmap,
    colour_curve: ColourCurve,
}

struct QuadSphere {
    object: Handle<Object>,
    /// Quad-sphere parameters necessary for quad mesh generation among other
    /// things.
    params: QuadSphereParams,
    /// A factor to apply to subdivision and collapse ranges. A higher value
    /// means better planetary detail, however the number of quads (and thus
    /// vertex and polygon count) scales in quadratic fashion to this value and
    /// should not be set too high.
    range_factor: f64,
    /// A vector storing the collapse ranges for each subdivision level.
    collapse_ranges: Vec<f64>,
    /// A vector storing the subdivision ranges for each subdivision level.
    subdivide_ranges: Vec<f64>,
    centre_pos: Vector3<f64>,
    centre_dist: f64,
    cull_sin_theta: f64,
    cull_cos_theta: f64,
    faces: Option<[Rc<RefCell<Quad>>; 6]>,
    normal_update_queue: RefCell<Vec<Rc<RefCell<Quad>>>>,
    quad_pool: QuadPool,

    task_manager: RefCell<TaskManager>,
}

impl QuadSphereParams {
    fn quad_length(&self, level: i32) -> i32 {
        (1 << (self.max_subdivision - level)) * self.quad_mesh_size
    }

    /// Converts a `VertCoord` to a position on the quad sphere (not
    /// normalized).
    fn vc_to_pos(&self, coord: VertCoord) -> Vector3<f64> {
        let (x, y, z) = match coord {
            VertCoord(Plane::XP, a, b) => (self.max_coord, b, self.max_coord - a),
            VertCoord(Plane::XN, a, b) => (0, b, a),
            VertCoord(Plane::YP, a, b) => (a, self.max_coord, self.max_coord - b),
            VertCoord(Plane::YN, a, b) => (a, 0, b),
            VertCoord(Plane::ZP, a, b) => (a, b, self.max_coord),
            VertCoord(Plane::ZN, a, b) => (self.max_coord - a, b, 0),
        };
        Vector3::new(-1.0 + x as f64 * 2.0 / self.max_coord as f64,
                     -1.0 + y as f64 * 2.0 / self.max_coord as f64,
                     -1.0 + z as f64 * 2.0 / self.max_coord as f64)
    }

    /// Converts a `VertCoord` to a position on the quad sphere in integer
    /// coordinates (not normalized).
    fn vc_to_ipos(&self, coord: VertCoord) -> (i32, i32, i32) {
        match coord {
            VertCoord(Plane::XP, a, b) => (self.max_coord, b, self.max_coord - a),
            VertCoord(Plane::XN, a, b) => (0, b, a),
            VertCoord(Plane::YP, a, b) => (a, self.max_coord, self.max_coord - b),
            VertCoord(Plane::YN, a, b) => (a, 0, b),
            VertCoord(Plane::ZP, a, b) => (a, b, self.max_coord),
            VertCoord(Plane::ZN, a, b) => (self.max_coord - a, b, 0),
        }
    }

    /// Converts integer coordinates to a `VertCoord` on the given plane.
    fn ipos_to_vc(&self, plane: Plane, x: i32, y: i32, z: i32) -> VertCoord {
        match plane {
            Plane::XP => VertCoord(Plane::XP, self.max_coord - z, y),
            Plane::XN => VertCoord(Plane::XN, z, y),
            Plane::YP => VertCoord(Plane::YP, x, self.max_coord - z),
            Plane::YN => VertCoord(Plane::YN, x, z),
            Plane::ZP => VertCoord(Plane::ZP, x, y),
            Plane::ZN => VertCoord(Plane::ZN, self.max_coord - x, y),
        }
    }

    fn sample_heightmap_avg(&self, coord: VertCoord) -> f32 {
        let (x, y, z) = self.vc_to_ipos(coord);

        let mut sum = 0.0;
        let mut count = 0;

        if x == self.max_coord {
            let vc = self.ipos_to_vc(Plane::XP, x, y, z);
            let height = self.sample_heightmap(vc);
            sum += height;
            count += 1;
        } else if x == 0 {
            let vc = self.ipos_to_vc(Plane::XN, x, y, z);
            let height = self.sample_heightmap(vc);
            sum += height;
            count += 1;
        }

        if y == self.max_coord {
            let vc = self.ipos_to_vc(Plane::YP, x, y, z);
            let height = self.sample_heightmap(vc);
            sum += height;
            count += 1;
        } else if y == 0 {
            let vc = self.ipos_to_vc(Plane::YN, x, y, z);
            let height = self.sample_heightmap(vc);
            sum += height;
            count += 1;
        }

        if z == self.max_coord {
            let vc = self.ipos_to_vc(Plane::ZP, x, y, z);
            let height = self.sample_heightmap(vc);
            sum += height;
            count += 1;
        } else if z == 0 {
            let vc = self.ipos_to_vc(Plane::ZN, x, y, z);
            let height = self.sample_heightmap(vc);
            sum += height;
            count += 1;
        }

        sum / count as f32
    }

    fn sample_heightmap(&self, coord: VertCoord) -> f32 {
        let VertCoord(plane, vc_a, vc_b) = coord;
        let x = vc_a as f32 / self.max_coord as f32;
        let y = vc_b as f32 / self.max_coord as f32;

        self.heightmap.sample(plane, x, y)
    }

    fn calc_quad_verts(&self,
                       plane: Plane,
                       base_coord: (i32, i32),
                       subdivision: i32)
                       -> QuadGenResult {
        let vert_step = 1 << (self.max_subdivision - subdivision);
        let adj_size = self.quad_mesh_size + 1;

        let mut vpos = Vec::with_capacity((adj_size * adj_size) as usize);
        let mut vcolour = Vec::with_capacity((adj_size * adj_size) as usize);
        for x in 0..adj_size {
            for y in 0..adj_size {
                let vert_coord = VertCoord(plane,
                                           base_coord.0 + x * vert_step,
                                           base_coord.1 + y * vert_step);
                let vert_pos = self.vc_to_pos(vert_coord).normalize();

                let height = self.sample_heightmap_avg(vert_coord) as f64;
                // Add some noise to make things look a bit more interesting
                let noise = self.generator
                    .get_value(vert_pos.x as f64, vert_pos.y as f64, vert_pos.z as f64);
                let height = noise * 0.25 + height * 0.75;
                let height = f64::min(1.0, f64::max(0.0, height));

                let (r, g, b, _) = self.colour_curve.get_colour(height);
                let r = (r as f32) / 255.0;
                let g = (g as f32) / 255.0;
                let b = (b as f32) / 255.0;

                let height = (self.radius + self.min_height) +
                             height * (self.max_height - self.min_height);
                let vert_pos = vert_pos * height;

                vpos.push(vert_pos.cast());
                vcolour.push(Vector3::new(r, g, b));
            }
        }


        let mut vnorm = Vec::with_capacity((adj_size * adj_size) as usize);
        for x in 0..adj_size {
            for y in 0..adj_size {
                let up = if y < adj_size - 1 {
                    y + 1
                } else {
                    y
                };
                let down = if y > 0 {
                    y - 1
                } else {
                    y
                };
                let left = if x > 0 {
                    x - 1
                } else {
                    x
                };
                let right = if x < adj_size - 1 {
                    x + 1
                } else {
                    x
                };

                let left = vpos[vert_off(left as u16, y as u16, adj_size as u16) as usize];
                let right = vpos[vert_off(right as u16, y as u16, adj_size as u16) as usize];
                let up = vpos[vert_off(x as u16, up as u16, adj_size as u16) as usize];
                let down = vpos[vert_off(x as u16, down as u16, adj_size as u16) as usize];

                vnorm.push((right - left).cross(up - down).normalize());
            }
        }

        QuadGenResult {
            vpos,
            vcolour,
            vnorm,
        }
    }
}

impl QuadSphere {
    fn new(scene: &mut Scene,
           quad_mesh_size: i32,
           max_subdivision: i32,
           radius: f64,
           min_height: f64,
           max_height: f64) -> QuadSphere {
        assert!(quad_mesh_size > 1 && (quad_mesh_size % 2) == 0 && quad_mesh_size <= 128);
        let bits = (quad_mesh_size - 1).leading_zeros();
        assert!(max_subdivision as u32 <= (bits - 2));

        let range_factor = 4.0;
        // Try to put a good upper bound on the number of quads:
        //
        //  * Assume quads are half the size they would normally be (accounts
        //    for changing quad density at cube edges)
        //
        //  * Assume all quads within collapse range are subdivided
        //
        //  * Multiply by max subdivision (subdivision level 0 can only have the
        //    6 root quads)
        let pool_size = range_factor * 2.0 * 1.6 * 2.0;
        let pool_size = std::f64::consts::PI * pool_size * pool_size;
        let pool_size = pool_size * max_subdivision as f64;
        let pool_size = pool_size as usize;

        let max_coord = (1 << max_subdivision) * quad_mesh_size;

        let heightmap = Heightmap::load("heightmap/");
        let mut colour_curve = ColourCurve::new();
        colour_curve.add_control_point(0.0, (0x42, 0x29, 0x13, 0xff));
        colour_curve.add_control_point(0.5, (0x58, 0x35, 0x17, 0xff));
        colour_curve.add_control_point(0.51, (0x5e, 0x36, 0x15, 0xff));
        colour_curve.add_control_point(0.6, (0x7c, 0x4e, 0x28, 0xff));
        colour_curve.add_control_point(0.8, (0x74, 0x44, 0x1d, 0xff));
        colour_curve.add_control_point(0.81, (0x8b, 0x59, 0x31, 0xff));
        colour_curve.add_control_point(1.0, (0x9a, 0x66, 0x3b, 0xff));

        let params1 = QuadSphereParams {
            quad_mesh_size: quad_mesh_size,
            max_subdivision: max_subdivision,
            max_coord: max_coord,
            radius: radius,
            min_height: min_height,
            max_height: max_height,
            generator: create_generator(),
            heightmap: heightmap.clone(),
            colour_curve: colour_curve.clone(),
        };
        let params2 = QuadSphereParams {
            quad_mesh_size: quad_mesh_size,
            max_subdivision: max_subdivision,
            max_coord: max_coord,
            radius: radius,
            min_height: min_height,
            max_height: max_height,
            generator: create_generator(),
            heightmap: heightmap,
            colour_curve: colour_curve,
        };

        let object = scene.create_object();
        let mut rv = QuadSphere {
            object: object,
            params: params1,
            range_factor: range_factor,
            collapse_ranges: Vec::new(),
            subdivide_ranges: Vec::new(),
            centre_pos: Vector3::unit_z(),
            centre_dist: 1.0,
            cull_sin_theta: 0.0,
            cull_cos_theta: 0.0,
            faces: None,
            normal_update_queue: RefCell::new(Vec::new()),
            quad_pool: QuadPool::new(scene, quad_mesh_size, pool_size),
            // TODO: placeholder value for batch size and lifetime, possibly
            // investigate tuning these value.
            task_manager: RefCell::new(TaskManager::new(100, 10, params2)),
        };

        rv.init(scene);
        rv
    }
    fn init(&mut self, scene: &mut Scene) {
        self.init_ranges();

        let xp_quad_result = self.params.calc_quad_verts(Plane::XP, (0, 0), 0);
        let xn_quad_result = self.params.calc_quad_verts(Plane::XN, (0, 0), 0);
        let yp_quad_result = self.params.calc_quad_verts(Plane::YP, (0, 0), 0);
        let yn_quad_result = self.params.calc_quad_verts(Plane::YN, (0, 0), 0);
        let zp_quad_result = self.params.calc_quad_verts(Plane::ZP, (0, 0), 0);
        let zn_quad_result = self.params.calc_quad_verts(Plane::ZN, (0, 0), 0);

        let xp_quad = self.quad_pool.get_quad(scene);
        let xn_quad = self.quad_pool.get_quad(scene);
        let yp_quad = self.quad_pool.get_quad(scene);
        let yn_quad = self.quad_pool.get_quad(scene);
        let zp_quad = self.quad_pool.get_quad(scene);
        let zn_quad = self.quad_pool.get_quad(scene);

        {
            let self_ptr = Rc::downgrade(&xp_quad);
            let mut xp_quad = xp_quad.borrow_mut();
            xp_quad.plane = Plane::XP;
            xp_quad.pos = QuadPos::None;
            xp_quad.unmerged_normals.copy_from_slice(&xp_quad_result.vnorm);
            xp_quad.patching_dirty = true;
            xp_quad.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
            xp_quad.self_ptr = Some(self_ptr);
            xp_quad.north = Some(Rc::downgrade(&yp_quad));
            xp_quad.south = Some(Rc::downgrade(&yn_quad));
            xp_quad.east = Some(Rc::downgrade(&zn_quad));
            xp_quad.west = Some(Rc::downgrade(&zp_quad));
            xp_quad.cur_subdivision = 0;
            xp_quad.base_coord = (0, 0);
            xp_quad.mid_coord_pos = xp_quad.mid_coord_pos(self);

            scene.set_object_parent(xp_quad.object, Some(self.object));
            let mesh = xp_quad.mesh.as_ref().unwrap();
            *mesh.vpos_mut(scene).unwrap() = xp_quad_result.vpos;
            *mesh.vcolour_mut(scene).unwrap() = xp_quad_result.vcolour;
            *mesh.vnorm_mut(scene).unwrap() = xp_quad_result.vnorm;
        }
        self.queue_normal_update(xp_quad.clone());

        {
            let self_ptr = Rc::downgrade(&xn_quad);
            let mut xn_quad = xn_quad.borrow_mut();
            xn_quad.plane = Plane::XN;
            xn_quad.pos = QuadPos::None;
            xn_quad.unmerged_normals.copy_from_slice(&xn_quad_result.vnorm);
            xn_quad.patching_dirty = true;
            xn_quad.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
            xn_quad.self_ptr = Some(self_ptr);
            xn_quad.north = Some(Rc::downgrade(&yp_quad));
            xn_quad.south = Some(Rc::downgrade(&yn_quad));
            xn_quad.east = Some(Rc::downgrade(&zp_quad));
            xn_quad.west = Some(Rc::downgrade(&zn_quad));
            xn_quad.cur_subdivision = 0;
            xn_quad.base_coord = (0, 0);
            xn_quad.mid_coord_pos = xn_quad.mid_coord_pos(self);

            scene.set_object_parent(xn_quad.object, Some(self.object));
            let mesh = xn_quad.mesh.as_ref().unwrap();
            *mesh.vpos_mut(scene).unwrap() = xn_quad_result.vpos;
            *mesh.vcolour_mut(scene).unwrap() = xn_quad_result.vcolour;
            *mesh.vnorm_mut(scene).unwrap() = xn_quad_result.vnorm;
        }
        self.queue_normal_update(xn_quad.clone());

        {
            let self_ptr = Rc::downgrade(&yp_quad);
            let mut yp_quad = yp_quad.borrow_mut();
            yp_quad.plane = Plane::YP;
            yp_quad.pos = QuadPos::None;
            yp_quad.unmerged_normals.copy_from_slice(&yp_quad_result.vnorm);
            yp_quad.patching_dirty = true;
            yp_quad.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
            yp_quad.self_ptr = Some(self_ptr);
            yp_quad.north = Some(Rc::downgrade(&zn_quad));
            yp_quad.south = Some(Rc::downgrade(&zp_quad));
            yp_quad.east = Some(Rc::downgrade(&xp_quad));
            yp_quad.west = Some(Rc::downgrade(&xn_quad));
            yp_quad.cur_subdivision = 0;
            yp_quad.base_coord = (0, 0);
            yp_quad.mid_coord_pos = yp_quad.mid_coord_pos(self);

            scene.set_object_parent(yp_quad.object, Some(self.object));
            let mesh = yp_quad.mesh.as_ref().unwrap();
            *mesh.vpos_mut(scene).unwrap() = yp_quad_result.vpos;
            *mesh.vcolour_mut(scene).unwrap() = yp_quad_result.vcolour;
            *mesh.vnorm_mut(scene).unwrap() = yp_quad_result.vnorm;
        }
        self.queue_normal_update(yp_quad.clone());

        {
            let self_ptr = Rc::downgrade(&yn_quad);
            let mut yn_quad = yn_quad.borrow_mut();
            yn_quad.plane = Plane::YN;
            yn_quad.pos = QuadPos::None;
            yn_quad.unmerged_normals.copy_from_slice(&yn_quad_result.vnorm);
            yn_quad.patching_dirty = true;
            yn_quad.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
            yn_quad.self_ptr = Some(self_ptr);
            yn_quad.north = Some(Rc::downgrade(&zp_quad));
            yn_quad.south = Some(Rc::downgrade(&zn_quad));
            yn_quad.east = Some(Rc::downgrade(&xp_quad));
            yn_quad.west = Some(Rc::downgrade(&xn_quad));
            yn_quad.cur_subdivision = 0;
            yn_quad.base_coord = (0, 0);
            yn_quad.mid_coord_pos = yn_quad.mid_coord_pos(self);

            scene.set_object_parent(yn_quad.object, Some(self.object));
            let mesh = yn_quad.mesh.as_ref().unwrap();
            *mesh.vpos_mut(scene).unwrap() = yn_quad_result.vpos;
            *mesh.vcolour_mut(scene).unwrap() = yn_quad_result.vcolour;
            *mesh.vnorm_mut(scene).unwrap() = yn_quad_result.vnorm;
        }
        self.queue_normal_update(yn_quad.clone());

        {
            let self_ptr = Rc::downgrade(&zp_quad);
            let mut zp_quad = zp_quad.borrow_mut();
            zp_quad.plane = Plane::ZP;
            zp_quad.pos = QuadPos::None;
            zp_quad.unmerged_normals.copy_from_slice(&zp_quad_result.vnorm);
            zp_quad.patching_dirty = true;
            zp_quad.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
            zp_quad.self_ptr = Some(self_ptr);
            zp_quad.north = Some(Rc::downgrade(&yp_quad));
            zp_quad.south = Some(Rc::downgrade(&yn_quad));
            zp_quad.east = Some(Rc::downgrade(&xp_quad));
            zp_quad.west = Some(Rc::downgrade(&xn_quad));
            zp_quad.cur_subdivision = 0;
            zp_quad.base_coord = (0, 0);
            zp_quad.mid_coord_pos = zp_quad.mid_coord_pos(self);

            scene.set_object_parent(zp_quad.object, Some(self.object));
            let mesh = zp_quad.mesh.as_ref().unwrap();
            *mesh.vpos_mut(scene).unwrap() = zp_quad_result.vpos;
            *mesh.vcolour_mut(scene).unwrap() = zp_quad_result.vcolour;
            *mesh.vnorm_mut(scene).unwrap() = zp_quad_result.vnorm;
        }
        self.queue_normal_update(zp_quad.clone());

        {
            let self_ptr = Rc::downgrade(&zn_quad);
            let mut zn_quad = zn_quad.borrow_mut();
            zn_quad.plane = Plane::ZN;
            zn_quad.pos = QuadPos::None;
            zn_quad.patching_dirty = true;
            zn_quad.unmerged_normals.copy_from_slice(&zn_quad_result.vnorm);
            zn_quad.needs_normal_merge = QUAD_SIDE_FLAGS_ALL;
            zn_quad.self_ptr = Some(self_ptr);
            zn_quad.north = Some(Rc::downgrade(&yp_quad));
            zn_quad.south = Some(Rc::downgrade(&yn_quad));
            zn_quad.east = Some(Rc::downgrade(&xn_quad));
            zn_quad.west = Some(Rc::downgrade(&xp_quad));
            zn_quad.cur_subdivision = 0;
            zn_quad.base_coord = (0, 0);
            zn_quad.mid_coord_pos = zn_quad.mid_coord_pos(self);

            scene.set_object_parent(zn_quad.object, Some(self.object));
            let mesh = zn_quad.mesh.as_ref().unwrap();
            *mesh.vpos_mut(scene).unwrap() = zn_quad_result.vpos;
            *mesh.vcolour_mut(scene).unwrap() = zn_quad_result.vcolour;
            *mesh.vnorm_mut(scene).unwrap() = zn_quad_result.vnorm;
        }
        self.queue_normal_update(zn_quad.clone());

        self.faces = Some([xp_quad, xn_quad, yp_quad, yn_quad, zp_quad, zn_quad]);
    }

    fn init_ranges(&mut self) {
        self.collapse_ranges = Vec::with_capacity(self.params.max_subdivision as usize + 1);
        self.subdivide_ranges = Vec::with_capacity(self.params.max_subdivision as usize + 1);

        for lvl in 0..(self.params.max_subdivision + 1) {
            let quad_length = self.quad_length(lvl);
            // Multiply by two since a plane ranges from -1.0 to +1.0
            let real_quad_length = 2.0 * (quad_length as f64 / self.params.max_coord as f64);

            // sqrt(0.5^2 + 1.5^2) = ~1.6, this means any point within a quad
            // will cause all four neighbours to be subdivided as well.
            let collapse_range = self.range_factor * 2.0 * 1.6 * real_quad_length;
            let subdivide_range = self.range_factor * 1.6 * real_quad_length;

            let r = 1.0;
            let collapse_cos_theta = f64::cos(f64::min(std::f64::consts::PI, collapse_range / r));
            let subdivide_cos_theta = f64::cos(f64::min(std::f64::consts::PI, subdivide_range / r));

            self.collapse_ranges.push(collapse_cos_theta);
            self.subdivide_ranges.push(subdivide_cos_theta);
        }
    }

    fn quad_mesh_size(&self) -> i32 {
        self.params.quad_mesh_size
    }

    fn max_subdivision(&self) -> i32 {
        self.params.max_subdivision
    }

    fn quad_length(&self, level: i32) -> i32 {
        self.params.quad_length(level)
    }

    fn centre_pos(&self) -> Vector3<f64> {
        self.centre_pos
    }

    /// Lookup the range required for us to try collapsing a quad for a given
    /// subdivision level. The returned value isn't a distance, but instead the
    /// cosine of the angle of the arc formed over that distance.
    fn collapse_range(&self, subdivision: i32) -> f64 {
        self.collapse_ranges[subdivision as usize]
    }

    /// Lookup the range required for us to try subdividing a quad for a given
    /// subdivision level. See `collapse_range()` for more details.
    fn subdivide_range(&self, subdivision: i32) -> f64 {
        self.subdivide_ranges[subdivision as usize]
    }

    fn calc_cull_range(&mut self) {
        let d = self.centre_dist;
        let r0 = self.params.radius + self.params.min_height;
        let r1 = self.params.radius + self.params.max_height;

        let a = f64::sqrt(d * d - r0 * r0);
        let b = f64::sqrt(r1 * r1 - r0 * r0);
        let e = a + b;

        let cos_theta = (e * e + d * d - r1 * r1) / 2.0 * d * r1;
        let sin_theta = f64::sqrt(1.0 - cos_theta * cos_theta);

        self.cull_sin_theta = sin_theta;
        self.cull_cos_theta = cos_theta;
    }

    fn horizon_cull_range(&self, angle_size: (f64, f64)) -> f64 {
        let (sin_alpha, cos_alpha) = angle_size;

        // Addition rule
        let range = self.cull_cos_theta * cos_alpha - self.cull_sin_theta * sin_alpha;
        let range = f64::max(0.0, range);

        range
    }

    fn queue_normal_update(&self, quad: Rc<RefCell<Quad>>) {
        self.normal_update_queue.borrow_mut().push(quad);
    }

    /// Converts a `VertCoord` to a position on the quad sphere (not
    /// normalized).
    fn vc_to_pos(&self, coord: VertCoord) -> Vector3<f64> {
        self.params.vc_to_pos(coord)
    }

    fn quad_pool(&self) -> &QuadPool {
        &self.quad_pool
    }

    fn task_manager(&self) -> RefMut<TaskManager> {
        self.task_manager.borrow_mut()
    }

    #[allow(dead_code)]
    fn debug_print_stats(&self) {
        let mut level_count = vec![0; self.params.max_subdivision as usize + 1];
        let mut visible_level_count = vec![0; self.params.max_subdivision as usize + 1];
        let mut horizon_cull_count = 0;
        let mut horizon_cull_visible_count = 0;

        for i in 0..6 {
            let q = self.faces.as_ref().unwrap()[i].clone();
            q.borrow().debug_get_stats(
                self,
                &mut level_count,
                &mut visible_level_count,
                &mut horizon_cull_count,
                &mut horizon_cull_visible_count);
        }

        println!("Horizon culled count = {} ({} total)",
                 horizon_cull_visible_count,
                 horizon_cull_count);
        println!("");

        println!("------");
        println!("Levels");
        println!("------");
        for (i, l) in level_count.iter().enumerate() {
            println!("Level {} = {}", i, l);
        }
        println!("");

        println!("--------------");
        println!("Visible levels");
        println!("--------------");
        for (i, l) in visible_level_count.iter().enumerate() {
            println!("Visible level {} = {}", i, l);
        }
        println!("");
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
            q.clone()
        } else {
            q.borrow().debug_find_quad(path)
        }
    }
}

impl QuadSphere {
    fn start(&mut self, _scene: &mut Scene) {
    }

    fn update(&mut self, scene: &mut Scene) {
        self.calc_cull_range();
        self.task_manager.borrow_mut().check_done();

        for i in 0..6 {
            let q = self.faces.as_ref().unwrap()[i].clone();
            q.borrow_mut().check_subdivision(self, scene);
        }

        for q in &*self.normal_update_queue.borrow() {
            let mut q = q.borrow_mut();
            if q.patching_dirty {
                q.update_normals(self, scene);
            }
            q.patching_dirty = false;
        }

        self.task_manager.borrow_mut().flush();
        self.task_manager.borrow_mut().dec_lifetime();

        for q in &*self.normal_update_queue.borrow() {
            let mut q = q.borrow_mut();
            if q.needs_normal_merge != QUAD_SIDE_FLAGS_NONE {
                q.update_edge_normals(self, scene);
            }
            q.needs_normal_merge = QUAD_SIDE_FLAGS_NONE;
        }

        self.normal_update_queue.borrow_mut().clear();
    }

    fn destroy(&mut self, scene: &mut Scene) {
        for i in 0..6 {
            let q = self.faces.as_ref().unwrap()[i].clone();
            q.borrow_mut().cleanup(scene);
            scene.destroy_object(q.borrow().object);
        }
        self.quad_pool.cleanup(scene);
    }
}

/// Manages subdivision tasks, allowing them to be executed asynchronously on a
/// separate thread and retrieved when complete. Tasks have a "lifetime"
/// associated with them and are forgotten if it runs out.
struct TaskManager {
    manager_rx: Receiver<TaskManagerMsg>,
    executor_tx: Sender<TaskExecutorMsg>,
    batch_size: usize,
    task_lifetime: i32,
    cur_batch: Vec<(TaskId, TaskData)>,
    tasks: HashMap<TaskId, (i32, Option<TaskResult>)>,
}

/// Structure containing the common data required to execute subdivision tasks.
struct TaskExecutor {
    manager_tx: Sender<TaskManagerMsg>,
    executor_rx: Receiver<TaskExecutorMsg>,
    params: QuadSphereParams,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
/// A type which represents a subdivision task, contains the base `VertCoord`
/// and subdivision level of the quad to be subdivided.
struct TaskId(VertCoord, i32);

#[derive(Copy, Clone)]
/// A type containing the data required for a subdivision task. Currently, no
/// data is passed.
struct TaskData(());

/// The result of a subdivision task.
struct TaskResult {
    q1: QuadGenResult,
    q2: QuadGenResult,
    q3: QuadGenResult,
    q4: QuadGenResult,
}

/// Describes the status of a task.
enum TaskStatus {
    /// The given task hasn't been started.
    NotStarted,
    /// The given task has been scheduled, but hasn't completed yet.
    NotComplete,
    /// The given task has been completed with the contained result.
    Complete(TaskResult),
}

/// Messages the `TaskManager` can receive.
enum TaskManagerMsg {
    /// Sent by the `TaskExecutor` to signal a batch job has been completed and
    /// pass the result back.
    BatchComplete(Vec<(TaskId, TaskResult)>),
    /// Sent by the `TaskExecutor` to signal it has received a `Stop` message
    /// and is about to terminate.
    Stop,
}

/// Messages the `TaskExecutor` can receive.
enum TaskExecutorMsg {
    /// Sent by the `TaskManager` to start a batch job.
    ExecBatch(Vec<(TaskId, TaskData)>),
    /// Sent by the `TaskManager` to request the `TaskExecutor` terminate.
    Stop,
}

impl TaskManager {
    /// Constructs a new `TaskManager` with the specified batch size, task
    /// lifetime and quad-sphere parameters.
    pub fn new(batch_size: usize, task_lifetime: i32, params: QuadSphereParams) -> TaskManager {
        let (executor_tx, executor_rx) = channel::<TaskExecutorMsg>();
        let (manager_tx, manager_rx) = channel::<TaskManagerMsg>();
        thread::spawn(move || {
            TaskManager::task_exec_thread(TaskExecutor {
                manager_tx,
                executor_rx,
                params,
            })
        });
        TaskManager {
            executor_tx,
            manager_rx,
            batch_size,
            task_lifetime,
            cur_batch: Vec::new(),
            tasks: HashMap::new(),
        }
    }

    /// Try to get the result of the specified task.
    pub fn get_task(&mut self, task_id: TaskId) -> TaskStatus {
        use std::collections::hash_map::Entry;
        match self.tasks.entry(task_id) {
            Entry::Occupied(e) => {
                if e.get().1.is_some() {
                    TaskStatus::Complete(e.remove().1.unwrap())
                } else {
                    TaskStatus::NotComplete
                }
            }
            Entry::Vacant(_) => TaskStatus::NotStarted,
        }
    }

    /// Reset the lifetime of a task back to the initial value, indicating this
    /// task is still needed. Does nothing if the task hasn't been started.
    pub fn reset_task_lifetime(&mut self, task_id: TaskId) {
        use std::collections::hash_map::Entry;
        match self.tasks.entry(task_id) {
            Entry::Occupied(mut e) => {
                // Reset task lifetime
                e.get_mut().0 = self.task_lifetime;
                return;
            }
            Entry::Vacant(_) => return,
        }
    }

    /// Schedules the given task to be executed, does nothing if the given task
    /// is currently executing or has been completed. The task is not sent for
    /// execution immediately, instead tasks are buffered and are only sent once
    /// a large enough batch size has been reached. To force buffered tasks to
    /// be executed now, call `flush`.
    pub fn add_task(&mut self, task_id: TaskId, task_data: TaskData) {
        use std::collections::hash_map::Entry;
        match self.tasks.entry(task_id) {
            Entry::Occupied(_) => return,
            Entry::Vacant(e) => {
                e.insert((self.task_lifetime, None));
            }
        }
        self.cur_batch.push((task_id, task_data));
        if self.cur_batch.len() >= self.batch_size {
            self.flush();
        }
    }

    /// Pushes the currently buffered tasks into a batch to be executed.
    pub fn flush(&mut self) {
        let mut batch = Vec::new();
        std::mem::swap(&mut batch, &mut self.cur_batch);
        self.executor_tx.send(TaskExecutorMsg::ExecBatch(batch)).unwrap();
    }

    /// Decrements the lifetime of all tasks, removes tasks which lifetimes
    /// reach zero.
    pub fn dec_lifetime(&mut self) {
        for v in self.tasks.values_mut() {
            v.0 -= 1;
        }
        self.tasks.retain(|_, v| v.0 > 0);
    }

    /// Check for batches completed by the `TaskExecutor`.
    pub fn check_done(&mut self) {
        loop {
            match self.manager_rx.try_recv() {
                Ok(TaskManagerMsg::BatchComplete(batch)) => {
                    for (task_id, result) in batch {
                        match self.tasks.get_mut(&task_id) {
                            Some(v) => v.1 = Some(result),
                            None => {
                                // We aren't interested in this result anymore
                                continue;
                            }
                        }
                    }
                },
                Ok(TaskManagerMsg::Stop) => panic!("Executor sent `Stop` message unexpectedly."),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => panic!("Executor thread disconnected unexpectedly."),
            }
        }
    }

    fn task_exec_thread(mut executor: TaskExecutor) {
        executor.execute();
    }
}

impl Drop for TaskManager {
    fn drop(&mut self) {
        self.executor_tx.send(TaskExecutorMsg::Stop).unwrap();
        loop {
            match self.manager_rx.recv() {
                Ok(TaskManagerMsg::Stop) => {
                    break;
                }
                Ok(_) => (),
                Err(_) => panic!("Executor thread disconnected unexpectedly."),
            }
        }
    }
}

impl TaskExecutor {
    fn execute(&mut self) {
        loop {
            match self.executor_rx.recv().unwrap() {
                TaskExecutorMsg::ExecBatch(batch) => {
                    let results = self.exec_batch(batch);
                    self.manager_tx.send(TaskManagerMsg::BatchComplete(results)).unwrap();
                },
                TaskExecutorMsg::Stop => {
                    self.manager_tx.send(TaskManagerMsg::Stop).unwrap();
                    break;
                }
            }
        }
    }

    fn exec_batch(&mut self, batch: Vec<(TaskId, TaskData)>) -> Vec<(TaskId, TaskResult)> {
        batch.into_iter().map(|(task_id, _)| {
            let TaskId(VertCoord(plane, base_coord_x, base_coord_y), subdivision) = task_id;
            let half_quad_length = self.params.quad_length(subdivision) / 2;

            let upper_left_base_coord = (base_coord_x, base_coord_y + half_quad_length);
            let upper_right_base_coord = (base_coord_x + half_quad_length,
                                          base_coord_y + half_quad_length);
            let lower_left_base_coord = (base_coord_x, base_coord_y);
            let lower_right_base_coord = (base_coord_x + half_quad_length, base_coord_y);
            let q1 = self.params.calc_quad_verts(plane, upper_left_base_coord, subdivision + 1);
            let q2 = self.params.calc_quad_verts(plane, upper_right_base_coord, subdivision + 1);
            let q3 = self.params.calc_quad_verts(plane, lower_left_base_coord, subdivision + 1);
            let q4 = self.params.calc_quad_verts(plane, lower_right_base_coord, subdivision + 1);
            let result = TaskResult {
                q1,
                q2,
                q3,
                q4,
            };
            (task_id, result)
        }).collect()
    }
}

struct CameraController {
    prev_time: std::time::Instant,
    camera_obj: Handle<Object>,
    /// Speed the camera moves in m/s
    speed: f32,
    /// Speed the camera rotates in degrees/s
    rot_speed: f32,
    mouse_prev_x: i32,
    mouse_prev_y: i32,
    cam_pos: Vector3<f32>,
    cam_rot: Quaternion<f32>,
}

impl CameraController {
    fn new(scene: &mut Scene) -> CameraController {
        let camera_obj = scene.create_object();
        let camera_far_obj = scene.create_object();
        let camera_near_obj = scene.create_object();
        scene.set_object_parent(camera_far_obj, Some(camera_obj));
        scene.set_object_parent(camera_near_obj, Some(camera_obj));

        let camera_far = scene.add_component::<Camera>(camera_far_obj).unwrap();
        let camera_near = scene.add_component::<Camera>(camera_near_obj).unwrap();
        camera_far.set_near_clip(scene, 100000.0).unwrap();
        camera_far.set_far_clip(scene, 10000000.0).unwrap();
        camera_far.set_order(scene, 0).unwrap();
        camera_near.set_near_clip(scene, 100.0).unwrap();
        camera_near.set_far_clip(scene, 110000.0).unwrap();
        camera_near.set_order(scene, 1).unwrap();

        camera_obj.set_world_pos(scene, -Vector3::unit_z() * 6_600_000.0).unwrap();

        CameraController {
            prev_time: std::time::Instant::now(),
            camera_obj: camera_obj,
            speed: 10000.0,
            rot_speed: 60.0,
            mouse_prev_x: 0,
            mouse_prev_y: 0,
            cam_pos: Vector3::zero(),
            cam_rot: Quaternion::one(),
        }
    }

    fn start(&mut self, scene: &mut Scene) {
        self.prev_time = std::time::Instant::now();

        let (mouse_x, mouse_y) = {
            let state = scene.mouse_state();
            (state.x(), state.y())
        };
        self.mouse_prev_x = mouse_x;
        self.mouse_prev_y = mouse_y;
    }

    fn update(&mut self, scene: &mut Scene) {
        let dt = self.prev_time.elapsed();
        let dt = dt.as_secs() as f32 + dt.subsec_nanos() as f32 / 1000000000.0;
        self.prev_time = std::time::Instant::now();

        let (forward_amount, left_amount, roll, speed_factor) = {
            let state = scene.keyboard_state();

            let mut forward = 0.0;
            let mut left = 0.0;
            let mut roll = 0.0;

            if state.is_scancode_pressed(Scancode::W) {
                forward += 1.0;
            }
            if state.is_scancode_pressed(Scancode::S) {
                forward -= 1.0;
            }

            if state.is_scancode_pressed(Scancode::A) {
                left += 1.0;
            }
            if state.is_scancode_pressed(Scancode::D) {
                left -= 1.0;
            }

            if state.is_scancode_pressed(Scancode::Q) {
                roll += 1.0;
            }
            if state.is_scancode_pressed(Scancode::E) {
                roll -= 1.0;
            }

            let speed_factor = if state.is_scancode_pressed(Scancode::LShift) {
                10.0
            } else {
                1.0
            };

            (forward, left, roll, speed_factor)
        };

        let (dx, dy) = {
            let state = scene.mouse_state();
            let dx = state.x() - self.mouse_prev_x;
            let dy = state.y() - self.mouse_prev_y;
            self.mouse_prev_x = state.x();
            self.mouse_prev_y = state.y();

            if state.right() { (dx, dy) } else { (0, 0) }
        };

        let roll = roll * self.rot_speed * dt;

        self.cam_rot = self.cam_rot * Quaternion::from(Euler {
            x: Deg(-dy as f32),
            y: Deg(-dx as f32),
            z: Deg(roll)
        });

        let old_pos = self.camera_obj.world_pos(scene).unwrap();

        let forward = Vector3::new(0.0, 0.0, -1.0) * forward_amount;
        let left = Vector3::new(-1.0, 0.0, 0.0) * left_amount;

        let direction = forward + left;

        let direction_world = self.cam_rot * direction * self.speed * speed_factor;

        self.cam_pos = old_pos + direction_world * dt;

        self.camera_obj.set_world_pos(scene, self.cam_pos).unwrap();
        self.camera_obj.set_world_rot(scene, self.cam_rot).unwrap();
    }

    fn cam_pos(&self) -> Vector3<f32> {
        self.cam_pos
    }

    fn destroy(&mut self, _scene: &mut Scene) {
    }
}

struct SunController {
    prev_time: std::time::Instant,
    /// Speed multiplier for the sun, 1.0 is a 24-hour day.
    speed: f32,
    sun_pos: Vector3<f32>,
    /// Whether the increase speed key was pressed last frame
    was_inc_key_down: bool,
    /// Whether the decrease speed key was pressed last frame
    was_dec_key_down: bool,
    sun_obj: Handle<Object>,
    sun_material: Handle<Material>,
    sun_cam_obj: Handle<Object>,
}

impl SunController {
    fn new(scene: &mut Scene) -> SunController {
        let sun_obj = scene.create_object();
        let mrenderer = scene.add_component::<MeshRenderer>(sun_obj).unwrap();
        let mesh = scene.create_mesh(6, 6);
        *mesh.vpos_mut(scene).unwrap() = vec![
            Vector3::new(-1.0, 1.0, 0.0),
            Vector3::new(-1.0, -1.0, 0.0),
            Vector3::new(1.0, -1.0, 0.0),
            Vector3::new(1.0, -1.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
            Vector3::new(-1.0, 1.0, 0.0),
        ];
        *mesh.indices_mut(scene).unwrap() = vec![
            0, 1, 2, 3, 4, 5
        ];
        mrenderer.set_mesh(scene, Some(mesh)).unwrap();
        // Render separately from the planet
        mrenderer.set_layers(scene, 2).unwrap();

        let shader = scene.create_shader(include_str!("sun_vs.glsl"),
                                         include_str!("sun_fs.glsl"),
                                         None);
        let material = scene.create_material(shader.clone()).unwrap();

        mrenderer.set_material(scene, Some(material.clone())).unwrap();

        let sun_cam_obj = scene.create_object();
        let sun_cam = scene.add_component::<Camera>(sun_cam_obj).unwrap();
        // Render behind the planet
        sun_cam.set_order(scene, -1).unwrap();
        sun_cam.set_near_clip(scene, 0.1).unwrap();
        sun_cam.set_far_clip(scene, 5.0).unwrap();
        sun_cam.set_layers(scene, 2).unwrap();

        let mut rv = SunController {
            prev_time: std::time::Instant::now(),
            speed: 1.0,
            sun_pos: Vector3::new(0.0, 1.0, 0.0),
            was_inc_key_down: false,
            was_dec_key_down: false,
            sun_obj: sun_obj,
            sun_material: material,
            sun_cam_obj: sun_cam_obj,
        };

        // 10 degrees
        rv.set_sun_size(scene, 3600.0 * 10.0);
        rv
    }

    fn set_sun_size(&mut self, scene: &mut Scene, arcsecs: f64) {
        let rad = (arcsecs / 3600.0).to_radians();
        let radius = f64::tan(0.5 * rad) as f32;
        self.sun_material
            .set_uniform(scene, "sun_radius", UniformValue::Float(radius))
            .unwrap();
    }

    fn update_sun(&mut self, scene: &mut Scene, camera_controller: &CameraController) {
        let up = Vector3::new(0.0, 1.0, 0.0);
        let cam_rot = camera_controller.cam_rot;
        let cam_up = cam_rot * up;

        self.sun_cam_obj.set_world_rot(scene, cam_rot).unwrap();

        let rot = Quaternion::look_at(-self.sun_pos, cam_up).invert();
        self.sun_obj.set_world_pos(scene, self.sun_pos.normalize()).unwrap();
        self.sun_obj.set_world_rot(scene, rot).unwrap();
    }

    fn start(&mut self, scene: &mut Scene, camera_controller: &CameraController) {
        self.prev_time = std::time::Instant::now();
        self.update_sun(scene, camera_controller);
    }

    fn update(&mut self, scene: &mut Scene, camera_controller: &CameraController) {
        let dt = self.prev_time.elapsed();
        let dt = dt.as_secs() as f32 + dt.subsec_nanos() as f32 / 1000000000.0;
        self.prev_time = std::time::Instant::now();

        let angle = (360.0 / (24.0 * 3600.0)) * dt * self.speed;
        let rot = Quaternion::from(Euler {
            x: Deg(angle),
            y: Deg(0.0),
            z: Deg(0.0),
        });
        self.sun_pos = rot * self.sun_pos;

        self.update_sun(scene, camera_controller);

        let (inc_key_down, dec_key_down) = {
            let state = scene.keyboard_state();
            let inc_key_down = state.is_scancode_pressed(Scancode::Period);
            let dec_key_down = state.is_scancode_pressed(Scancode::Comma);

            (inc_key_down, dec_key_down)
        };

        if inc_key_down && !self.was_inc_key_down {
            self.speed *= 2.0;
            println!("Increasing speed to {}", self.speed);
        }
        if dec_key_down && !self.was_dec_key_down {
            self.speed *= 0.5;
            println!("Decreasing speed to {}", self.speed);
        }
        self.was_inc_key_down = inc_key_down;
        self.was_dec_key_down = dec_key_down;
    }

    fn destroy(&mut self, _scene: &mut Scene) {
    }
}

impl Quad {
    /// Get the direct (same subdivision level) neighbouring quad on the given
    /// side if it exists, otherwise returns `None`.
    fn get_direct_side(&self, side: QuadSide) -> Option<Rc<RefCell<Quad>>> {
        match side {
            QuadSide::North => self.direct_north(),
            QuadSide::South => self.direct_south(),
            QuadSide::East => self.direct_east(),
            QuadSide::West => self.direct_west(),
        }
    }

    /// Get the neighbouring quad on the given side. The given quad will be of a
    /// subdivision level which is guaranteed to exist and not change; thus the
    /// returned quad could be of the same subdivision level or one subdivision
    /// level lower.
    fn get_side(&self, side: QuadSide) -> Rc<RefCell<Quad>> {
        match side {
            QuadSide::North => self.north(),
            QuadSide::South => self.south(),
            QuadSide::East => self.east(),
            QuadSide::West => self.west(),
        }
    }

    /// Merge the normals of this quad and the neighbouring quad(s) on the given
    /// `side`. Currently changes the corners but incorrectly, this is
    /// inconsequential however.
    fn merge_side_normals(&mut self, sphere: &QuadSphere, scene: &mut Scene, side: QuadSide) {
        // To simplify the problem we can first look at the situation as being
        // one of three states:
        //
        //   * No direct neighbour
        //
        //     +--------+----------------+
        //     |        |                |
        //     |  SELF  |                |
        //     |        |                |
        //     +--------+     OTHER      |
        //              |                |
        //              |                |
        //              |                |
        //              +----------------+
        //
        //  * Direct neighbour, not subdivided
        //
        //    +--------+--------+
        //    |        |        |
        //    |  SELF  | OTHER  |
        //    |        |        |
        //    +--------+--------+
        //
        //  * Direct neighbour, subdivided
        //
        //    +----------------+--------+
        //    |                |        |
        //    |                |   Q1   |
        //    |                |        |
        //    |      SELF      +--------+
        //    |                |        |
        //    |                |   Q2   |
        //    |                |        |
        //    +----------------+--------+
        //
        // In each case we can view the problem as trying to find for all pairs
        // of applicable quads to merge a "base" position and a "step"
        // vector. We calculate these initially assuming that all quads lie on
        // the same plane; then we use the function `map_vec_pos()` to to
        // convert these if necessary.
        //
        // This function purposefully doesn't handle the problem of quads
        // corners as this simplifies the logic.
        let quad_mesh_size = sphere.quad_mesh_size();
        let direct_side = self.get_direct_side(side);
        let (dir_sx, dir_sy) = match side {
            QuadSide::North => (1, 0),
            QuadSide::South => (1, 0),
            QuadSide::East => (0, 1),
            QuadSide::West => (0, 1),
        };
        if let Some(direct_side) = direct_side {
            let mut direct_side = direct_side.borrow_mut();
            if direct_side.is_subdivided() {
                let max = quad_mesh_size;
                let half = quad_mesh_size / 2;
                let ((base_sx, base_sy), (base_dx, base_dy)) = match side {
                    QuadSide::North => ((0, max), (0, 0)),
                    QuadSide::South => ((0, 0), (0, max)),
                    QuadSide::East => ((max, 0), (0, 0)),
                    QuadSide::West => ((0, 0), (max, 0)),
                };

                let (q1_pos, q2_pos) = match side {
                    QuadSide::North => (QuadPos::SouthWest, QuadPos::SouthEast),
                    QuadSide::South => (QuadPos::NorthWest, QuadPos::NorthEast),
                    QuadSide::East => (QuadPos::SouthWest, QuadPos::NorthWest),
                    QuadSide::West => (QuadPos::SouthEast, QuadPos::NorthEast),
                };

                let q1_pos = map_quad_pos(q1_pos, self.plane, direct_side.plane);
                let q2_pos = map_quad_pos(q2_pos, self.plane, direct_side.plane);

                let q1 = direct_side.get_child(q1_pos);
                let q2 = direct_side.get_child(q2_pos);

                let (step_sx, step_sy) = (dir_sx, dir_sy);
                let (step_dx, step_dy) = (dir_sx * 2, dir_sy * 2);

                let ((step_dx, step_dy), (base_dx, base_dy)) = map_vec_pos((step_dx, step_dy),
                                                                           (base_dx, base_dy),
                                                                           quad_mesh_size,
                                                                           self.plane,
                                                                           direct_side.plane);

                // TODO: don't touch corners
                self.merge_normals(sphere,
                                   scene,
                                   &mut *q1.borrow_mut(),
                                   (base_sx, base_sy),
                                   (step_sx, step_sy),
                                   (base_dx, base_dy),
                                   (step_dx, step_dy),
                                   quad_mesh_size / 2 + 1);

                let (base_sx2, base_sy2) = (base_sx + step_sx * half, base_sy + step_sy * half);

                // TODO: don't touch corners
                self.merge_normals(sphere,
                                   scene,
                                   &mut *q2.borrow_mut(),
                                   (base_sx2, base_sy2),
                                   (step_sx, step_sy),
                                   (base_dx, base_dy),
                                   (step_dx, step_dy),
                                   quad_mesh_size / 2 + 1);
            } else {
                let max = quad_mesh_size;
                let ((base_sx, base_sy), (base_dx, base_dy)) = match side {
                    QuadSide::North => ((0, max), (0, 0)),
                    QuadSide::South => ((0, 0), (0, max)),
                    QuadSide::East => ((max, 0), (0, 0)),
                    QuadSide::West => ((0, 0), (max, 0)),
                };

                let (step_sx, step_sy) = (dir_sx, dir_sy);
                let (step_dx, step_dy) = (dir_sx, dir_sy);

                let ((step_dx, step_dy), (base_dx, base_dy)) = map_vec_pos((step_dx, step_dy),
                                                                           (base_dx, base_dy),
                                                                           quad_mesh_size,
                                                                           self.plane,
                                                                           direct_side.plane);

                // TODO: don't touch corners
                self.merge_normals(sphere,
                                   scene,
                                   &mut *direct_side,
                                   (base_sx, base_sy),
                                   (step_sx, step_sy),
                                   (base_dx, base_dy),
                                   (step_dx, step_dy),
                                   quad_mesh_size + 1);
            }
        } else {
            let indirect_side = self.get_side(side);
            let mut indirect_side = indirect_side.borrow_mut();
            // TODO: copy paste code
            let (x, y) = match self.pos {
                QuadPos::NorthWest => (0, 1),
                QuadPos::NorthEast => (1, 1),
                QuadPos::SouthWest => (0, 0),
                QuadPos::SouthEast => (1, 0),
                QuadPos::None => {
                    // We know we are subdivided at least once so this should be
                    // unreachable.
                    panic!("`pos` cannot be QuadPos::None.")
                }
            };

            let max = quad_mesh_size;
            let half = quad_mesh_size / 2;
            let ((base_sx, base_sy), (base_dx, base_dy)) = match side {
                QuadSide::North => ((0, max), (x * half, 0)),
                QuadSide::South => ((0, 0), (x * half, max)),
                QuadSide::East => ((max, 0), (0, y * half)),
                QuadSide::West => ((0, 0), (max, y * half)),
            };
            let (step_sx, step_sy) = (dir_sx * 2, dir_sy * 2);
            let (step_dx, step_dy) = (dir_sx, dir_sy);

            let ((step_dx, step_dy), (base_dx, base_dy)) = map_vec_pos((step_dx, step_dy),
                                                                       (base_dx, base_dy),
                                                                       quad_mesh_size,
                                                                       self.plane,
                                                                       indirect_side.plane);

            // TODO: don't touch corners
            self.merge_normals(sphere,
                               scene,
                               &mut *indirect_side,
                               (base_sx, base_sy),
                               (step_sx, step_sy),
                               (base_dx, base_dy),
                               (step_dx, step_dy),
                               quad_mesh_size / 2 + 1);
        }
    }

    /// Merges the normals of two quads given base coordinates and step vectors
    /// for each.
    fn merge_normals(&mut self,
                     sphere: &QuadSphere,
                     scene: &mut Scene,
                     other: &mut Quad,
                     (base_sx, base_sy): (i32, i32),
                     (step_sx, step_sy): (i32, i32),
                     (base_dx, base_dy): (i32, i32),
                     (step_dx, step_dy): (i32, i32),
                     vert_count: i32) {
        let quad_mesh_size = sphere.quad_mesh_size();
        let adj_size = quad_mesh_size + 1;
        let vert_off = |x, y| vert_off(x, y, adj_size as u16);

        let mut self_normalized = Vec::new();
        let mut other_normalized = Vec::new();

        let self_mesh = self.mesh.as_ref().unwrap();
        let other_mesh = other.mesh.as_ref().unwrap();
        std::mem::swap(self_mesh.vnorm_mut(scene).unwrap(), &mut self_normalized);
        std::mem::swap(other_mesh.vnorm_mut(scene).unwrap(), &mut other_normalized);

        for i in 0..vert_count {
            let (sx, sy) = (base_sx + step_sx * i, base_sy + step_sy * i);
            let (dx, dy) = (base_dx + step_dx * i, base_dy + step_dy * i);
            let svert_off = vert_off(sx as u16, sy as u16) as usize;
            let dvert_off = vert_off(dx as u16, dy as u16) as usize;

            let combined = self.unmerged_normals[svert_off] + other.unmerged_normals[dvert_off];
            let normalized = combined.normalize();
            self_normalized[svert_off] = normalized;
            other_normalized[dvert_off] = normalized;
        }

        std::mem::swap(self_mesh.vnorm_mut(scene).unwrap(), &mut self_normalized);
        std::mem::swap(other_mesh.vnorm_mut(scene).unwrap(), &mut other_normalized);
    }

    /// Merge the normals on the specified corner of this quad and the other
    /// quads which share the given `corner`.
    ///
    /// # Panics
    ///
    /// Panics if `corner` is `QuadPos::None`
    fn merge_corner_normal(&mut self, sphere: &QuadSphere, scene: &mut Scene, corner: QuadPos) {
        // To understand the problem we can first look at the situation as being
        // one of six states:
        //
        //   * Direct vertical neighbour, direct horizontal neighbour
        //
        //              +--------+
        //              |        |
        //         Q3?  |   Q1   |
        //              |        |
        //     +--------%--------+
        //     |        |        |
        //     |   Q2   |  SELF  |
        //     |        |        |
        //     +--------+--------+
        //
        //   * Indirect vertical neighbour (midpoint), direct horizontal
        //     neighbour
        //
        //     +-----------------+
        //     |                 |
        //     |                 |
        //     |                 |
        //     |       Q1        |
        //     |                 |
        //     |                 |
        //     |                 |
        //     +--------%--------+
        //     |        |        |
        //     |   Q2   |  SELF  |
        //     |        |        |
        //     +--------+--------+
        //
        //   * Indirect vertical neighbour (corner), direct horizontal neighbour
        //
        //              +-----------------+
        //              |                 |
        //              |                 |
        //              |                 |
        //              |       Q1        |
        //              |                 |
        //         Q3?  |                 |
        //              |                 |
        //     +--------%--------+--------+
        //     |        |        |
        //     |   Q2   |  SELF  |
        //     |        |        |
        //     +--------+--------+
        //
        //   * Direct vertical neighbour, indirect horizontal neighbour
        //     (midpoint)
        //
        //     +-----------------+--------+
        //     |                 |        |
        //     |                 |   Q1   |
        //     |                 |        |
        //     |       Q2        %--------+
        //     |                 |        |
        //     |                 |  SELF  |
        //     |                 |        |
        //     +-----------------+--------+
        //
        //   * Direct vertical neighbour, indirect horizontal neighbour
        //     (corner)
        //
        //                       +--------+
        //                       |        |
        //                  Q3?  |   Q1   |
        //                       |        |
        //     +-----------------%--------+
        //     |                 |        |
        //     |                 |  SELF  |
        //     |                 |        |
        //     |       Q2        +--------+
        //     |                 |
        //     |                 |
        //     |                 |
        //     +-----------------+
        //
        //   * Indirect vertical neighbour (corner), indirect horizontal
        //     neighbour (corner)
        //
        //                       +-----------------+
        //                       |                 |
        //                       |                 |
        //                       |                 |
        //             Q3?       |       Q1        |
        //                       |                 |
        //                       |                 |
        //                       |                 |
        //     +-----------------%--------+--------+
        //     |                 |        |
        //     |                 |  SELF  |
        //     |                 |        |
        //     |       Q2        +--------+
        //     |                 |
        //     |                 |
        //     |                 |
        //     +-----------------+
        assert!(corner != QuadPos::None);

        let quad_mesh_size = sphere.quad_mesh_size();
        let adj_size = quad_mesh_size + 1;
        let vert_off = |x, y| vert_off(x, y, adj_size as u16);
        let max = quad_mesh_size;
        let half = quad_mesh_size / 2;

        let (side1, side2) = corner.split();

        let (q1, q1_is_direct) = match self.get_direct_side(side1) {
            Some(q1) => (q1, true),
            None => (self.get_side(side1), false),
        };
        let (q2, q2_is_direct) = match self.get_direct_side(side2) {
            Some(q2) => (q2, true),
            None => (self.get_side(side2), false),
        };

        let q1_plane = q1.borrow().plane;
        let q2_plane = q2.borrow().plane;

        let (self_x, self_y) = match corner {
            QuadPos::NorthWest => (0, max),
            QuadPos::NorthEast => (max, max),
            QuadPos::SouthWest => (0, 0),
            QuadPos::SouthEast => (max, 0),
            QuadPos::None => unreachable!(),
        };

        let q1_x = if q1_is_direct {
            match side2 {
                QuadSide::East => max,
                QuadSide::West => 0,
                _ => unreachable!(),
            }
        } else {
            // Note: we don't need to handle `QuadPos::None` here since we know we
            // aren't of the lowest subdivision level, otherwise `get_direct_side()` would
            // have returned `Some(_)`.
            let (_, self_hside) = self.pos.split();
            match (self_hside, side2) {
                (QuadSide::West, QuadSide::West) => 0,
                (QuadSide::West, QuadSide::East) => half,
                (QuadSide::East, QuadSide::West) => half,
                (QuadSide::East, QuadSide::East) => max,
                _ => unreachable!(),
            }
        };
        let q1_y = match side1 {
            QuadSide::North => 0,
            QuadSide::South => max,
            _ => unreachable!(),
        };
        let q1_pos = match corner {
            QuadPos::NorthWest => QuadPos::SouthWest,
            QuadPos::NorthEast => QuadPos::SouthEast,
            QuadPos::SouthWest => QuadPos::NorthWest,
            QuadPos::SouthEast => QuadPos::NorthEast,
            QuadPos::None => unreachable!(),
        };

        let (_, (q1_x, q1_y)) =
            map_vec_pos((0, 0), (q1_x, q1_y), quad_mesh_size, self.plane, q1_plane);
        let q1_pos = map_quad_pos(q1_pos, self.plane, q1_plane);

        let q2_x = match side2 {
            QuadSide::East => 0,
            QuadSide::West => max,
            _ => unreachable!(),
        };
        let q2_y = if q2_is_direct {
            match side1 {
                QuadSide::North => max,
                QuadSide::South => 0,
                _ => unreachable!(),
            }
        } else {
            // Note: we don't need to handle `QuadPos::None` here since we know we
            // aren't of the lowest subdivision level, otherwise `get_direct_side()` would
            // have returned `Some(_)`.
            let (self_vside, _) = self.pos.split();
            match (self_vside, side1) {
                (QuadSide::South, QuadSide::South) => 0,
                (QuadSide::South, QuadSide::North) => half,
                (QuadSide::North, QuadSide::South) => half,
                (QuadSide::North, QuadSide::North) => max,
                _ => unreachable!(),
            }
        };
        let q2_pos = match corner {
            QuadPos::NorthWest => QuadPos::NorthEast,
            QuadPos::NorthEast => QuadPos::NorthWest,
            QuadPos::SouthWest => QuadPos::SouthEast,
            QuadPos::SouthEast => QuadPos::SouthWest,
            QuadPos::None => unreachable!(),
        };

        let (_, (q2_x, q2_y)) =
            map_vec_pos((0, 0), (q2_x, q2_y), quad_mesh_size, self.plane, q2_plane);
        let q2_pos = map_quad_pos(q2_pos, self.plane, q2_plane);

        let q1 = if q1.borrow().is_subdivided() {
            q1.borrow().get_child(q1_pos).clone()
        } else {
            q1
        };
        let q2 = if q2.borrow().is_subdivided() {
            q2.borrow().get_child(q2_pos).clone()
        } else {
            q2
        };

        if self.plane != q1_plane && self.plane != q2_plane && q1_plane != q2_plane ||
           q1_x == half || q1_y == half || q2_x == half ||
           q2_y == half {
            // There will only be three quads to consider in the following two
            // cases:
            //
            //   * All quads are on different planes
            //   * A corner is on the mid-point of a quad edge

            let mut q1 = q1.borrow_mut();
            let mut q2 = q2.borrow_mut();

            let self_idx = vert_off(self_x as u16, self_y as u16) as usize;
            let q1_idx = vert_off(q1_x as u16, q1_y as u16) as usize;
            let q2_idx = vert_off(q2_x as u16, q2_y as u16) as usize;

            let combined =
                self.unmerged_normals[self_idx] +
                q1.unmerged_normals[q1_idx] +
                q2.unmerged_normals[q2_idx];
            let normalized = combined.normalize();

            self.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[self_idx] = normalized;
            q1.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[q1_idx] = normalized;
            q2.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[q2_idx] = normalized;
        } else {
            let side2 = map_quad_side(side2, self.plane, q1_plane);

            let mut q3 = q1.borrow()
                .get_direct_side(side2)
                .unwrap_or_else(|| q1.borrow().get_side(side2));
            let q3_plane = q3.borrow().plane;
            let q3_pos = corner.opposite();
            let q3_pos = map_quad_pos(q3_pos, self.plane, q3_plane);

            let (q3_x, q3_y) = match q3_pos {
                QuadPos::NorthWest => (0, max),
                QuadPos::NorthEast => (max, max),
                QuadPos::SouthWest => (0, 0),
                QuadPos::SouthEast => (max, 0),
                QuadPos::None => unreachable!(),
            };

            loop {
                let q3_next = match q3.borrow().get_child_opt(q3_pos) {
                    Some(q3_next) => q3_next.clone(),
                    None => break,
                };
                q3 = q3_next;
            }

            let mut q1 = q1.borrow_mut();
            let mut q2 = q2.borrow_mut();
            let mut q3 = q3.borrow_mut();

            let self_idx = vert_off(self_x as u16, self_y as u16) as usize;
            let q1_idx = vert_off(q1_x as u16, q1_y as u16) as usize;
            let q2_idx = vert_off(q2_x as u16, q2_y as u16) as usize;
            let q3_idx = vert_off(q3_x as u16, q3_y as u16) as usize;

            let combined =
                self.unmerged_normals[self_idx] +
                q1.unmerged_normals[q1_idx] +
                q2.unmerged_normals[q2_idx] +
                q3.unmerged_normals[q3_idx];
            let normalized = combined.normalize();

            self.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[self_idx] = normalized;
            q1.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[q1_idx] = normalized;
            q2.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[q2_idx] = normalized;
            q3.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[q3_idx] = normalized;
        }
    }
}

struct QuadPool {
    shader: Handle<Shader>,
    quad_material: Handle<Material>,
    indices_configs: Vec<Vec<u16>>,
    vertices_cap: usize,
    indices_cap: usize,
    pool: RefCell<Vec<Rc<RefCell<Quad>>>>,
    in_use: Cell<usize>,
}

impl QuadPool {
    fn new(scene: &mut Scene, quad_mesh_size: i32, initial_size: usize) -> QuadPool {
        let shader = scene.create_shader(include_str!("default_vs.glsl"),
                                         include_str!("default_fs.glsl"),
                                         None);
        let quad_material = scene.create_material(shader.clone()).unwrap();

        let mut indices_configs = Vec::with_capacity(16);
        for i in 0..16 {
            indices_configs.push(gen_indices(quad_mesh_size as u16, QuadSideFlags::from_bits(i).unwrap()));
        }

        let adj_size = quad_mesh_size as usize + 1;
        let vertices_cap = adj_size * adj_size;
        // Base the mesh indices capacity from `QUAD_SIDE_FLAGS_NONE` since that
        // generates the largest buffer size.
        let indices_cap = indices_configs[QUAD_SIDE_FLAGS_NONE.to_idx()].len();

        let pool = QuadPool {
            shader: shader,
            quad_material: quad_material,
            indices_configs: indices_configs,
            vertices_cap: vertices_cap,
            indices_cap: indices_cap,
            pool: RefCell::new(Vec::new()),
            in_use: Cell::new(0),
        };

        let mut pool_vec = Vec::new();
        for _ in 0..initial_size {
            let q = pool.create_quad(scene);
            pool_vec.push(q);
        }

        *pool.pool.borrow_mut() = pool_vec;

        pool
    }

    fn create_quad(&self, scene: &mut Scene) -> Rc<RefCell<Quad>> {
        let q = Rc::new(RefCell::new(Quad::new(scene)));

        {
            let mut q_borrow = q.borrow_mut();
            let mrenderer = scene.add_component::<MeshRenderer>(q_borrow.object).unwrap();
            let mesh = scene.create_mesh(self.vertices_cap, self.indices_cap);

            q_borrow.unmerged_normals = vec![Vector3::zero(); self.vertices_cap];
            // Don't allocate for vpos since it will be allocated elsewhere
            *mesh.vpos_mut(scene).unwrap() = Vec::new();
            *mesh.vnorm_mut(scene).unwrap() = vec![Vector3::zero(); self.vertices_cap];
            *mesh.indices_mut(scene).unwrap() = self.indices_configs[QUAD_SIDE_FLAGS_NONE.to_idx()].clone();
            mrenderer.set_enabled(scene, false).unwrap();
            mrenderer.set_mesh(scene, Some(mesh.clone())).unwrap();
            mrenderer.set_material(scene, Some(self.quad_material.clone())).unwrap();
            q_borrow.mrenderer = Some(mrenderer);
            q_borrow.mesh = Some(mesh);
        }

        q
    }

    /// Performs a partial default initialization of the given `Quad`
    fn default_init_quad(quad: &mut Quad) {
        quad.plane = Plane::XP;
        quad.pos = QuadPos::None;
        quad.base_coord = (0, 0);
        quad.cur_subdivision = 0;
        quad.mid_coord_pos = Vector3::new(0.0, 0.0, 0.0);
        quad.patch_flags = QUAD_SIDE_FLAGS_NONE;

        quad.patching_dirty = false;
        quad.needs_normal_merge = QUAD_SIDE_FLAGS_NONE;
        quad.render = false;

        quad.self_ptr = None;
        quad.children = None;
        quad.north = None;
        quad.south = None;
        quad.east = None;
        quad.west = None;
    }

    /// Get a quad from the pool, allocating a new one only if there are no more
    /// quads remaining in the pool.
    fn get_quad(&self, scene: &mut Scene) -> Rc<RefCell<Quad>> {
        if let Some(q) = self.pool.borrow_mut().pop() {
            QuadPool::default_init_quad(&mut *q.borrow_mut());
            self.in_use.set(self.in_use.get() + 1);
            q
        } else {
            let q = self.create_quad(scene);
            QuadPool::default_init_quad(&mut *q.borrow_mut());
            self.in_use.set(self.in_use.get() + 1);
            q
        }
    }

    /// Get the indices for a given `QuadSideFlags` configuration.
    fn get_indices(&self, flags: QuadSideFlags) -> &Vec<u16> {
        &self.indices_configs[flags.to_idx()]
    }

    /// Release ownership of the given quad and add it to the pool.
    fn recycle_quad(&self, scene: &mut Scene, quad: Rc<RefCell<Quad>>) {
        quad.borrow().mrenderer.as_ref().unwrap().set_enabled(scene, false).unwrap();
        quad.borrow_mut().render = false;
        self.pool.borrow_mut().push(quad);
        self.in_use.set(self.in_use.get() - 1);
    }

    #[allow(dead_code)]
    fn debug_print_stats(&self) {
        let in_use = self.in_use.get();
        let total = in_use + self.pool.borrow().len();
        println!("In use = {} / {}", in_use, total);
    }

    /// Perform cleanup in preparation for being destroyed.
    fn cleanup(&self, scene: &mut Scene) {
        scene.destroy_material(self.quad_material);
        scene.destroy_shader(self.shader);
        for q in &*self.pool.borrow() {
            scene.destroy_object(q.borrow().object);
        }
        self.pool.borrow_mut().clear();
    }
}

struct MainDriver {
    camera_controller: Option<CameraController>,
    sun_controller: Option<SunController>,
    quad_sphere: Option<QuadSphere>,
    skybox_obj: Option<Handle<Object>>,
    skybox_cam_obj: Option<Handle<Object>>,
}

struct MainDriverBehaviour(RefCell<MainDriver>);

impl BehaviourMessages for MainDriverBehaviour {
    fn start(&self, scene: &mut Scene) {
        self.0.borrow_mut().start(scene);
    }

    fn update(&self, scene: &mut Scene) {
        self.0.borrow_mut().update(scene);
    }

    fn destroy(&self, scene: &mut Scene) {
        self.0.borrow_mut().destroy(scene);
    }
}

impl MainDriver {
    fn start(&mut self, scene: &mut Scene) {
        self.camera_controller = Some(CameraController::new(scene));
        self.sun_controller = Some(SunController::new(scene));
        self.quad_sphere = Some(QuadSphere::new(scene, 8, 12, 6_000_000.0, 0.0, 160_000.0));

        let skybox_obj = scene.create_object();
        let skybox_renderer = scene.add_component::<MeshRenderer>(skybox_obj).unwrap();
        let skybox_mesh = scene.create_mesh(36, 36);
        *skybox_mesh.vpos_mut(scene).unwrap() = vec![
            // ZP
            Vector3::new(1.0, -1.0, 1.0),
            Vector3::new(-1.0, -1.0, 1.0),
            Vector3::new(-1.0, 1.0, 1.0),
            Vector3::new(-1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, -1.0, 1.0),
            // XP
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(1.0, -1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(1.0, 1.0, -1.0),
            Vector3::new(1.0, -1.0, -1.0),
            // ZN
            Vector3::new(-1.0, -1.0, -1.0),
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(1.0, 1.0, -1.0),
            Vector3::new(1.0, 1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(-1.0, -1.0, -1.0),
            // XN
            Vector3::new(-1.0, -1.0, 1.0),
            Vector3::new(-1.0, -1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(-1.0, 1.0, 1.0),
            Vector3::new(-1.0, -1.0, 1.0),
            // YN
            Vector3::new(1.0, -1.0, -1.0),
            Vector3::new(-1.0, -1.0, -1.0),
            Vector3::new(-1.0, -1.0, 1.0),
            Vector3::new(-1.0, -1.0, 1.0),
            Vector3::new(1.0, -1.0, 1.0),
            Vector3::new(1.0, -1.0, -1.0),
            // YP
            Vector3::new(1.0, 1.0, 1.0),
            Vector3::new(-1.0, 1.0, 1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(-1.0, 1.0, -1.0),
            Vector3::new(1.0, 1.0, -1.0),
            Vector3::new(1.0, 1.0, 1.0),
        ];
        *skybox_mesh.indices_mut(scene).unwrap() = vec![
            0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 11,
            12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35,
        ];
        let skybox_shader = scene.create_shader(
            include_str!("skybox_vs.glsl"),
            include_str!("skybox_fs.glsl"),
            None);
        let skybox_material = scene.create_material(skybox_shader.clone()).unwrap();
        skybox_renderer.set_material(scene, Some(skybox_material.clone())).unwrap();
        skybox_renderer.set_mesh(scene, Some(skybox_mesh)).unwrap();
        skybox_renderer.set_layers(scene, 4).unwrap();

        let skybox_cam_obj = scene.create_object();
        let skybox_cam = scene.add_component::<Camera>(skybox_cam_obj).unwrap();
        skybox_cam.set_near_clip(scene, 0.1).unwrap();
        skybox_cam.set_far_clip(scene, 5.0).unwrap();
        skybox_cam.set_layers(scene, 4).unwrap();
        skybox_cam.set_order(scene, -2).unwrap();

        let skybox_xp = load_image("skybox/xp.png").expect("Failed to load XP skybox");
        let skybox_xn = load_image("skybox/xn.png").expect("Failed to load XN skybox");
        let skybox_yp = load_image("skybox/yp.png").expect("Failed to load YP skybox");
        let skybox_yn = load_image("skybox/yn.png").expect("Failed to load YN skybox");
        let skybox_zp = load_image("skybox/zp.png").expect("Failed to load ZP skybox");
        let skybox_zn = load_image("skybox/zn.png").expect("Failed to load ZN skybox");

        let cubemap = scene.create_cubemap(skybox_zn.1 as usize,
                                           skybox_zn.2 as usize,
                                           [&skybox_xp.0,
                                            &skybox_xn.0,
                                            &skybox_yp.0,
                                            &skybox_yn.0,
                                            &skybox_zp.0,
                                            &skybox_zn.0]);

        skybox_material.set_uniform(scene, "cubemap", UniformValue::Cubemap(cubemap)).unwrap();

        self.skybox_obj = Some(skybox_obj);
        self.skybox_cam_obj = Some(skybox_cam_obj);

        self.camera_controller.as_mut().unwrap().start(scene);
        self.sun_controller.as_mut().unwrap().start(scene, self.camera_controller.as_ref().unwrap());
        self.quad_sphere.as_mut().unwrap().start(scene);
    }

    fn update(&mut self, scene: &mut Scene) {
        self.camera_controller.as_mut().unwrap().update(scene);
        self.sun_controller.as_mut().unwrap().update(scene, self.camera_controller.as_ref().unwrap());


        {
            let quad_sphere = self.quad_sphere.as_mut().unwrap();
            let cam_pos = self.camera_controller.as_ref().unwrap().cam_pos();
            let sun_pos = self.sun_controller.as_ref().unwrap().sun_pos;
            quad_sphere.quad_pool.quad_material
                .set_uniform(scene, "light_dir", UniformValue::Vec3(sun_pos.into()))
                .unwrap();

            quad_sphere.centre_pos = cam_pos.normalize().cast();
            quad_sphere.centre_dist = cam_pos.magnitude() as f64;
            quad_sphere.update(scene);
        }

        let sun_pos = self.sun_controller.as_ref().unwrap().sun_pos;
        let cam_rot = self.camera_controller.as_ref().unwrap().cam_rot;
        let skybox_rot = Quaternion::between_vectors(Vector3::unit_z(), sun_pos);

        self.skybox_cam_obj
            .as_ref()
            .unwrap()
            .set_world_rot(scene, cam_rot)
            .unwrap();

        self.skybox_obj
            .as_ref()
            .unwrap()
            .set_world_rot(scene, skybox_rot)
            .unwrap();
    }

    fn destroy(&mut self, scene: &mut Scene) {
        self.camera_controller.as_mut().unwrap().destroy(scene);
        self.sun_controller.as_mut().unwrap().destroy(scene);
        self.quad_sphere.as_mut().unwrap().destroy(scene);
    }
}


fn main() {
    let sdl = sdl2::init().expect("Failed to initialize SDL2");
    let title = format!("PlanetGen {}", env!("CARGO_PKG_VERSION"));

    let mut scene = Scene::new(sdl, &title);
    scene.create_behaviour(MainDriverBehaviour(RefCell::new(MainDriver {
        camera_controller: None,
        sun_controller: None,
        quad_sphere: None,
        skybox_obj: None,
        skybox_cam_obj: None,
    })));

    loop {
        if !scene.do_frame() {
            break;
        }
    }
}
