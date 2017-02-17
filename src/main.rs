#[macro_use]
extern crate bitflags;
extern crate cgmath;
extern crate glium;
extern crate num;
extern crate planetgen_engine;

mod gen;

use std::cell::RefCell;
use std::rc::{Rc, Weak};

use cgmath::{Deg, Euler, InnerSpace, Rotation, Quaternion, Vector3};

use glium::DisplayBuild;

use num::{Zero, One};

use planetgen_engine::{Behaviour, BehaviourMessages, Camera, Material, Mesh, Scene, Shader};

use gen::{gen_indices, gen_vertices, vert_off, PatchFlags, PATCH_FLAGS_NONE, PATCH_FLAGS_NORTH,
          PATCH_FLAGS_SOUTH, PATCH_FLAGS_EAST, PATCH_FLAGS_WEST};

impl From<QuadSide> for PatchFlags {
    fn from(side: QuadSide) -> Self {
        match side {
            QuadSide::North => PATCH_FLAGS_NORTH,
            QuadSide::South => PATCH_FLAGS_SOUTH,
            QuadSide::East => PATCH_FLAGS_EAST,
            QuadSide::West => PATCH_FLAGS_WEST,
        }
    }
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
    NorthWest,
    NorthEast,
    SouthWest,
    SouthEast,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuadSide {
    North,
    South,
    East,
    West,
}

impl QuadPos {
    fn to_idx(&self) -> usize {
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
    fn split(self) -> (QuadSide, QuadSide) {
        match self {
            QuadPos::NorthWest => (QuadSide::North, QuadSide::West),
            QuadPos::NorthEast => (QuadSide::North, QuadSide::East),
            QuadPos::SouthWest => (QuadSide::South, QuadSide::West),
            QuadPos::SouthEast => (QuadSide::South, QuadSide::East),
            QuadPos::None => panic!("Cannot call `split` on QuadPos::None.")
        }
    }

    /// Returns the position opposite to `pos`.
    fn opposite(self) -> QuadPos {
        match self {
            QuadPos::NorthWest => QuadPos::SouthEast,
            QuadPos::NorthEast => QuadPos::SouthWest,
            QuadPos::SouthWest => QuadPos::NorthEast,
            QuadPos::SouthEast => QuadPos::NorthWest,
            QuadPos::None => panic!("Cannot call `opposite` on QuadPos::None.")
        }
    }

}

fn calc_plane_mapping<T: num::Signed>(src_plane: Plane, dst_plane: Plane) -> ((T, T), (T, T), (T, T)) {
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
        (YP, XP) => ((zero(), one()),  (zero(), -one()), (one(),  zero())),
        (YP, XN) => ((one(),  zero()), (zero(), one()),  (-one(), zero())),
        (YP, ZN) => ((one(),  one()),  (-one(), zero()), (zero(), -one())),
        (XP, YP) => ((one(),  zero()), (zero(), one()),  (-one(), zero())),
        (XN, YP) => ((zero(), one()),  (zero(), -one()), (one(),  zero())),
        (ZN, YP) => ((one(),  one()),  (-one(), zero()), (zero(), -one())),
        // YN
        (YN, XP) => ((one(),  zero()), (zero(), one()),  (-one(), zero())),
        (YN, XN) => ((zero(), one()),  (zero(), -one()), (one(),  zero())),
        (YN, ZN) => ((one(),  one()),  (-one(), zero()), (zero(), -one())),
        (XP, YN) => ((zero(), one()),  (zero(), -one()), (one(),  zero())),
        (XN, YN) => ((one(),  zero()), (zero(), one()),  (-one(), zero())),
        (ZN, YN) => ((one(),  one()),  (-one(), zero()), (zero(), -one())),
        _ => {
            ((zero(), zero()), (one(), zero()), (zero(), one()))
        }
    }
}

fn translate_quad_pos(pos: QuadPos, src_plane: Plane, dst_plane: Plane) -> QuadPos {
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

fn translate_quad_side(side: QuadSide, src_plane: Plane, dst_plane: Plane) -> QuadSide {
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

#[test]
fn test_quad_pos_translate() {
    assert_eq!(translate_quad_pos(QuadPos::NorthWest, Plane::XP, Plane::XP), QuadPos::NorthWest);
    assert_eq!(translate_quad_pos(QuadPos::NorthEast, Plane::XP, Plane::XP), QuadPos::NorthEast);
    assert_eq!(translate_quad_pos(QuadPos::SouthWest, Plane::XP, Plane::XP), QuadPos::SouthWest);
    assert_eq!(translate_quad_pos(QuadPos::SouthEast, Plane::XP, Plane::XP), QuadPos::SouthEast);

    assert_eq!(translate_quad_pos(QuadPos::NorthWest, Plane::XP, Plane::ZN), QuadPos::NorthWest);
    assert_eq!(translate_quad_pos(QuadPos::NorthEast, Plane::XP, Plane::ZN), QuadPos::NorthEast);
    assert_eq!(translate_quad_pos(QuadPos::SouthWest, Plane::XP, Plane::ZN), QuadPos::SouthWest);
    assert_eq!(translate_quad_pos(QuadPos::SouthEast, Plane::XP, Plane::ZN), QuadPos::SouthEast);

    assert_eq!(translate_quad_pos(QuadPos::NorthWest, Plane::XP, Plane::YP), QuadPos::SouthWest);
    assert_eq!(translate_quad_pos(QuadPos::NorthEast, Plane::XP, Plane::YP), QuadPos::NorthWest);
    // lower left
    assert_eq!(translate_quad_pos(QuadPos::SouthEast, Plane::XP, Plane::YP), QuadPos::NorthEast);

    assert_eq!(translate_quad_pos(QuadPos::SouthWest, Plane::XP, Plane::YP), QuadPos::SouthEast);
    assert_eq!(translate_quad_pos(QuadPos::SouthWest, Plane::ZN, Plane::YP), QuadPos::NorthEast);
    assert_eq!(translate_quad_pos(QuadPos::SouthWest, Plane::XN, Plane::YP), QuadPos::NorthWest);

    assert_eq!(translate_quad_pos(QuadPos::NorthWest, Plane::XP, Plane::YN), QuadPos::NorthEast);
    assert_eq!(translate_quad_pos(QuadPos::NorthWest, Plane::ZN, Plane::YN), QuadPos::SouthEast);
    assert_eq!(translate_quad_pos(QuadPos::NorthWest, Plane::XN, Plane::YN), QuadPos::SouthWest);

    assert_eq!(translate_quad_pos(QuadPos::SouthEast, Plane::ZN, Plane::YP), QuadPos::NorthWest);
}

#[test]
fn test_quad_side_translate() {
    assert_eq!(translate_quad_side(QuadSide::North, Plane::YP, Plane::ZP), QuadSide::North);
    assert_eq!(translate_quad_side(QuadSide::West, Plane::YP, Plane::XP), QuadSide::North);
    assert_eq!(translate_quad_side(QuadSide::South, Plane::YP, Plane::ZN), QuadSide::North);
    assert_eq!(translate_quad_side(QuadSide::East, Plane::YP, Plane::XN), QuadSide::North);
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
    mid_coord_pos: Vector3<f32>,
    patch_flags: PatchFlags,

    non_normalized: Vec<Vector3<f32>>,

    /// True if this quad needs the `non_normalized` normals to be recomputed.
    needs_normal_update: bool,
    /// True if this quad needs to have quad normals to be merge with
    /// neighbouring quads.
    needs_normal_merge: bool,

    /// Pointer to this quad
    self_ptr: Option<Weak<RefCell<Quad>>>,
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
        let indices = gen_indices(sphere.quad_mesh_size, PATCH_FLAGS_NONE);

        let vert_off = |x, y| vert_off(x, y, adj_size);
        for x in 0..adj_size {
            for y in 0..adj_size {
                let vert_coord = VertCoord(self.plane,
                                           self.base_coord.0 + x as u32 * vert_step,
                                           self.base_coord.1 + y as u32 * vert_step);

                let vert_pos = map_vertcoord(vert_coord, sphere.max_coord).normalize();
                let off = vert_off(x, y);
                vertices[off as usize] = Vector3::new(vert_pos.x, vert_pos.y, vert_pos.z);
            }
        }

        self.non_normalized.resize(vertices.len(), Vector3::zero());

        // Base the mesh indices from `PATCH_FLAGS_NONE` since that generates the
        // largest buffer size.
        let mesh = scene.create_mesh(vertices.len(), indices.len());
        *mesh.vpos_mut(scene).unwrap() = vertices;
        *mesh.indices_mut(scene).unwrap() = indices;

        let shader = scene.create_shader(
            include_str!("default_vs.glsl"),
            include_str!("default_fs.glsl"),
            None);
        let material = scene.create_material(shader.clone()).unwrap();

        self.mesh = Some(mesh);
        self.shader = Some(shader);
        self.material = Some(material);

        self.calc_normals(scene);

        self.mid_coord_pos = self.mid_coord_pos(sphere);

        self.patch_flags = PATCH_FLAGS_NONE;

        // TODO: reduce cloning
        let self_object = self.behaviour.object(scene).unwrap().clone();
        let sphere_object = sphere.behaviour().object(scene).unwrap().clone();
        scene.set_object_parent(&self_object, Some(&sphere_object));
    }

    fn calc_normals(&mut self, scene: &mut Scene) {
        {
            let verts = self.mesh.as_ref().unwrap().vpos(scene).unwrap();
            let indices = self.mesh.as_ref().unwrap().indices(scene).unwrap();

            for n in &mut self.non_normalized {
                *n = Vector3::new(0.0, 0.0, 0.0);
            }

            let mut i = 0;
            while i < indices.len() {
                let i = {
                    let tmp = i;
                    i += 3;
                    tmp
                };
                let v1 = verts[indices[i] as usize];
                let v2 = verts[indices[i + 1] as usize];
                let v3 = verts[indices[i + 2] as usize];
                let a = v2 - v1;
                let b = v3 - v1;
                let surf_normal = a.cross(b);
                self.non_normalized[indices[i] as usize] += surf_normal;
                self.non_normalized[indices[i + 1] as usize] += surf_normal;
                self.non_normalized[indices[i + 2] as usize] += surf_normal;
            }
        }

        let normals = self.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap();
        normals.clear();
        for n in &self.non_normalized {
            normals.push(n.normalize());
        }
    }

    /// Calculates the coordinates of the middle of this quad.
    fn mid_coord_pos(&self, sphere: &QuadSphere) -> Vector3<f32> {
        let half_quad_length = sphere.quad_length(self.cur_subdivision) / 2;
        let mid_coord = (self.base_coord.0 + half_quad_length, self.base_coord.1 + half_quad_length);
        map_vertcoord(VertCoord(self.plane, mid_coord.0, mid_coord.1), sphere.max_coord).normalize()
    }

    fn in_subdivision_range(&self, sphere: &QuadSphere) -> bool {
        if self.cur_subdivision == sphere.max_subdivision {
            return false
        }

        let range = sphere.subdivide_range(self.cur_subdivision);

        let centre_pos = sphere.centre_pos();
        let cur_range = centre_pos.dot(self.mid_coord_pos);

        // Note: comparison may seem swapped since lower values mean a greater
        // angle / arc.
        cur_range >= range
    }

    fn in_collapse_range(&self, sphere: &QuadSphere) -> bool {
        let range = sphere.collapse_range(self.cur_subdivision);

        let centre_pos = sphere.centre_pos();
        let cur_range = centre_pos.dot(self.mid_coord_pos);

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
                let pos = translate_quad_pos(QuadPos::SouthWest, self.plane, north_borrow.plane);
                north_borrow.get_child_opt(pos)
                    .map(Rc::clone)
            },
            QuadPos::NorthEast => {
                let north = self.north();
                let north_borrow = north.borrow();
                let pos = translate_quad_pos(QuadPos::SouthEast, self.plane, north_borrow.plane);
                north_borrow.get_child_opt(pos)
                    .map(Rc::clone)
            },
        }
    }

    fn direct_south(&self) -> Option<Rc<RefCell<Quad>>> {
        match self.pos {
            QuadPos::NorthWest | QuadPos::NorthEast | QuadPos::None => Some(self.south()),
            QuadPos::SouthWest => {
                let south = self.south();
                let south_borrow = south.borrow();
                let pos = translate_quad_pos(QuadPos::NorthWest, self.plane, south_borrow.plane);
                south_borrow.get_child_opt(pos)
                    .map(Rc::clone)
            },
            QuadPos::SouthEast => {
                let south = self.south();
                let south_borrow = south.borrow();
                let pos = translate_quad_pos(QuadPos::NorthEast, self.plane, south_borrow.plane);
                south_borrow.get_child_opt(pos)
                    .map(Rc::clone)
            },
        }
    }

    fn direct_east(&self) -> Option<Rc<RefCell<Quad>>> {
        match self.pos {
            QuadPos::NorthWest | QuadPos::SouthWest | QuadPos::None => Some(self.east()),
            QuadPos::NorthEast => {
                let east = self.east();
                let east_borrow = east.borrow();
                let pos = translate_quad_pos(QuadPos::NorthWest, self.plane, east_borrow.plane);
                east_borrow.get_child_opt(pos)
                    .map(Rc::clone)
            },
            QuadPos::SouthEast => {
                let east = self.east();
                let east_borrow = east.borrow();
                let pos = translate_quad_pos(QuadPos::SouthWest, self.plane, east_borrow.plane);
                east_borrow.get_child_opt(pos)
                    .map(Rc::clone)
            },
        }
    }

    fn direct_west(&self) -> Option<Rc<RefCell<Quad>>> {
        match self.pos {
            QuadPos::NorthEast | QuadPos::SouthEast | QuadPos::None => Some(self.west()),
            QuadPos::NorthWest => {
                let west = self.west();
                let west_borrow = west.borrow();
                let pos = translate_quad_pos(QuadPos::NorthEast, self.plane, west_borrow.plane);
                west_borrow.get_child_opt(pos)
                    .map(Rc::clone)
            },
            QuadPos::SouthWest => {
                let west = self.west();
                let west_borrow = west.borrow();
                let pos = translate_quad_pos(QuadPos::SouthEast, self.plane, west_borrow.plane);
                west_borrow.get_child_opt(pos)
                    .map(Rc::clone)
            },
        }
    }

    fn is_subdivided(&self) -> bool {
        self.children.is_some()
    }

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

    /// Checks if this quad is able to collapse without causing quad-tree
    /// invariants to be violated.
    fn can_collapse(&self) -> bool {
        assert!(self.is_subdivided(), "can_collapse() should only be called on subdivided quads");
        let direct_north = self.direct_north().unwrap();
        let direct_south = self.direct_south().unwrap();
        let direct_east = self.direct_east().unwrap();
        let direct_west = self.direct_west().unwrap();

        let direct_north = direct_north.borrow();
        if direct_north.is_subdivided() {
            let pos1 = translate_quad_pos(QuadPos::SouthWest, self.plane, direct_north.plane);
            let pos2 = translate_quad_pos(QuadPos::SouthEast, self.plane, direct_north.plane);
            let q1 = direct_north.get_child(pos1);
            let q2 = direct_north.get_child(pos2);
            if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                return false
            }
        }

        let direct_south = direct_south.borrow();
        if direct_south.is_subdivided() {
            let pos1 = translate_quad_pos(QuadPos::NorthWest, self.plane, direct_south.plane);
            let pos2 = translate_quad_pos(QuadPos::NorthEast, self.plane, direct_south.plane);
            let q1 = direct_south.get_child(pos1);
            let q2 = direct_south.get_child(pos2);
            if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                return false
            }
        }

        let direct_east = direct_east.borrow();
        if direct_east.is_subdivided() {
            let pos1 = translate_quad_pos(QuadPos::NorthWest, self.plane, direct_east.plane);
            let pos2 = translate_quad_pos(QuadPos::SouthWest, self.plane, direct_east.plane);
            let q1 = direct_east.get_child(pos1);
            let q2 = direct_east.get_child(pos2);
            if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                return false
            }
        }

        let direct_west = direct_west.borrow();
        if direct_west.is_subdivided() {
            let pos1 = translate_quad_pos(QuadPos::NorthEast, self.plane, direct_west.plane);
            let pos2 = translate_quad_pos(QuadPos::SouthEast, self.plane, direct_west.plane);
            let q1 = direct_west.get_child(pos1);
            let q2 = direct_west.get_child(pos2);
            if q1.borrow().is_subdivided() || q2.borrow().is_subdivided() {
                return false
            }
        }

        for q in self.children.as_ref().unwrap() {
            if q.borrow().is_subdivided() {
                return false
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
            let self_ptr = Rc::downgrade(&upper_left);
            let mut upper_left = upper_left.borrow_mut();
            upper_left.plane = self.plane;
            upper_left.pos = QuadPos::NorthWest;
            upper_left.needs_normal_update = true;
            upper_left.needs_normal_merge = true;
            upper_left.self_ptr = Some(self_ptr);
            upper_left.north = direct_north.as_ref().map(Rc::downgrade);
            upper_left.east = Some(Rc::downgrade(&upper_right));
            upper_left.south = Some(Rc::downgrade(&lower_left));
            upper_left.west = direct_west.as_ref().map(Rc::downgrade);
            upper_left.cur_subdivision = self.cur_subdivision + 1;
            upper_left.base_coord = (self.base_coord.0, self.base_coord.1 + half_quad_length);
            upper_left.init(sphere, scene);
        }
        sphere.queue_normal_update(upper_left.clone());

        {
            let self_ptr = Rc::downgrade(&upper_right);
            let mut upper_right = upper_right.borrow_mut();
            upper_right.plane = self.plane;
            upper_right.pos = QuadPos::NorthEast;
            upper_right.needs_normal_update = true;
            upper_right.needs_normal_merge = true;
            upper_right.self_ptr = Some(self_ptr);
            upper_right.north = direct_north.as_ref().map(Rc::downgrade);
            upper_right.east = direct_east.as_ref().map(Rc::downgrade);
            upper_right.south = Some(Rc::downgrade(&lower_right));
            upper_right.west = Some(Rc::downgrade(&upper_left));
            upper_right.cur_subdivision = self.cur_subdivision + 1;
            upper_right.base_coord = (self.base_coord.0 + half_quad_length, self.base_coord.1 + half_quad_length);
            upper_right.init(sphere, scene);
        }
        sphere.queue_normal_update(upper_right.clone());

        {
            let self_ptr = Rc::downgrade(&lower_left);
            let mut lower_left = lower_left.borrow_mut();
            lower_left.plane = self.plane;
            lower_left.pos = QuadPos::SouthWest;
            lower_left.needs_normal_update = true;
            lower_left.needs_normal_merge = true;
            lower_left.self_ptr = Some(self_ptr);
            lower_left.north = Some(Rc::downgrade(&upper_left));
            lower_left.east = Some(Rc::downgrade(&lower_right));
            lower_left.south = direct_south.as_ref().map(Rc::downgrade);
            lower_left.west = direct_west.as_ref().map(Rc::downgrade);
            lower_left.cur_subdivision = self.cur_subdivision + 1;
            lower_left.base_coord = (self.base_coord.0, self.base_coord.1);
            lower_left.init(sphere, scene);
        }
        sphere.queue_normal_update(lower_left.clone());

        {
            let self_ptr = Rc::downgrade(&lower_right);
            let mut lower_right = lower_right.borrow_mut();
            lower_right.plane = self.plane;
            lower_right.pos = QuadPos::SouthEast;
            lower_right.needs_normal_update = true;
            lower_right.needs_normal_merge = true;
            lower_right.self_ptr = Some(self_ptr);
            lower_right.north = Some(Rc::downgrade(&upper_right));
            lower_right.east = direct_east.as_ref().map(Rc::downgrade);
            lower_right.south = direct_south.as_ref().map(Rc::downgrade);
            lower_right.west = Some(Rc::downgrade(&lower_left));
            lower_right.cur_subdivision = self.cur_subdivision + 1;
            lower_right.base_coord = (self.base_coord.0 + half_quad_length, self.base_coord.1);
            lower_right.init(sphere, scene);
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
            let pos1 = translate_quad_pos(QuadPos::SouthWest, self.plane, north_borrow.plane);
            let pos2 = translate_quad_pos(QuadPos::SouthEast, self.plane, north_borrow.plane);
            let q1 = north_borrow.get_child(pos1);
            let q2 = north_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags = PatchFlags::from(translate_quad_side(QuadSide::South, self.plane, north_borrow.plane));
            q1_borrow.patch_flags &= !flags;
            q1_borrow.needs_normal_update = true;
            q1_borrow.needs_normal_merge = true;
            q2_borrow.patch_flags &= !flags;
            q2_borrow.needs_normal_update = true;
            q2_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut north_borrow = direct_north.borrow_mut();
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            north_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(direct_north.clone());
        }

        if south_subdivided {
            let south_borrow = direct_south.borrow();
            let pos1 = translate_quad_pos(QuadPos::NorthWest, self.plane, south_borrow.plane);
            let pos2 = translate_quad_pos(QuadPos::NorthEast, self.plane, south_borrow.plane);
            let q1 = south_borrow.get_child(pos1);
            let q2 = south_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags = PatchFlags::from(translate_quad_side(QuadSide::North, self.plane, south_borrow.plane));
            q1_borrow.patch_flags &= !flags;
            q1_borrow.needs_normal_update = true;
            q1_borrow.needs_normal_merge = true;
            q2_borrow.patch_flags &= !flags;
            q2_borrow.needs_normal_update = true;
            q2_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut south_borrow = direct_south.borrow_mut();
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            south_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(direct_south.clone());
        }

        if east_subdivided {
            let east_borrow = direct_east.borrow();
            let pos1 = translate_quad_pos(QuadPos::NorthWest, self.plane, east_borrow.plane);
            let pos2 = translate_quad_pos(QuadPos::SouthWest, self.plane, east_borrow.plane);
            let q1 = east_borrow.get_child(pos1);
            let q2 = east_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags = PatchFlags::from(translate_quad_side(QuadSide::West, self.plane, east_borrow.plane));
            q1_borrow.patch_flags &= !flags;
            q1_borrow.needs_normal_update = true;
            q1_borrow.needs_normal_merge = true;
            q2_borrow.patch_flags &= !flags;
            q2_borrow.needs_normal_update = true;
            q2_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut east_borrow = direct_east.borrow_mut();
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            east_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(direct_east.clone());
        }

        if west_subdivided {
            let west_borrow = direct_west.borrow();
            let pos1 = translate_quad_pos(QuadPos::NorthEast, self.plane, west_borrow.plane);
            let pos2 = translate_quad_pos(QuadPos::SouthEast, self.plane, west_borrow.plane);
            let q1 = west_borrow.get_child(pos1);
            let q2 = west_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags = PatchFlags::from(translate_quad_side(QuadSide::East, self.plane, west_borrow.plane));
            q1_borrow.patch_flags &= !flags;
            q1_borrow.needs_normal_update = true;
            q1_borrow.needs_normal_merge = true;
            q2_borrow.patch_flags &= !flags;
            q2_borrow.needs_normal_update = true;
            q2_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut west_borrow = direct_west.borrow_mut();
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            west_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(direct_west.clone());
        }

        {
            let mut upper_left = upper_left.borrow_mut();
            let mut upper_right = upper_right.borrow_mut();
            let mut lower_left = lower_left.borrow_mut();
            let mut lower_right = lower_right.borrow_mut();

            if !north_subdivided {
                upper_left.patch_flags |= PATCH_FLAGS_NORTH;
                upper_right.patch_flags |= PATCH_FLAGS_NORTH;
            }

            if !south_subdivided {
                lower_left.patch_flags |= PATCH_FLAGS_SOUTH;
                lower_right.patch_flags |= PATCH_FLAGS_SOUTH;
            }

            if !east_subdivided {
                upper_right.patch_flags |= PATCH_FLAGS_EAST;
                lower_right.patch_flags |= PATCH_FLAGS_EAST;
            }

            if !west_subdivided {
                upper_left.patch_flags |= PATCH_FLAGS_WEST;
                lower_left.patch_flags |= PATCH_FLAGS_WEST;
            }
        }

        self.needs_normal_update = false;
        self.needs_normal_merge = false;

        self.children = Some([upper_left, upper_right, lower_left, lower_right]);
    }

    fn collapse(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        for q in self.children.as_ref().unwrap() {
            // TODO: reduce cloning
            let q_obj = q.borrow().behaviour().object(scene).unwrap().clone();
            scene.destroy_object(&q_obj);
        }

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
            let pos1 = translate_quad_pos(QuadPos::SouthWest, self.plane, north_borrow.plane);
            let pos2 = translate_quad_pos(QuadPos::SouthEast, self.plane, north_borrow.plane);
            let q1 = north_borrow.get_child(pos1);
            let q2 = north_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags = PatchFlags::from(translate_quad_side(QuadSide::South, self.plane, north_borrow.plane));
            q1_borrow.patch_flags |= flags;
            q1_borrow.needs_normal_update = true;
            q1_borrow.needs_normal_merge = true;
            q2_borrow.patch_flags |= flags;
            q2_borrow.needs_normal_update = true;
            q2_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut north_borrow = direct_north.borrow_mut();
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            north_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(direct_north.clone());
        }

        if south_subdivided {
            let south_borrow = direct_south.borrow();
            let pos1 = translate_quad_pos(QuadPos::NorthWest, self.plane, south_borrow.plane);
            let pos2 = translate_quad_pos(QuadPos::NorthEast, self.plane, south_borrow.plane);
            let q1 = south_borrow.get_child(pos1);
            let q2 = south_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags = PatchFlags::from(translate_quad_side(QuadSide::North, self.plane, south_borrow.plane));
            q1_borrow.patch_flags |= flags;
            q1_borrow.needs_normal_update = true;
            q1_borrow.needs_normal_merge = true;
            q2_borrow.patch_flags |= flags;
            q2_borrow.needs_normal_update = true;
            q2_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut south_borrow = direct_south.borrow_mut();
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            south_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(direct_south.clone());
        }

        if east_subdivided {
            let east_borrow = direct_east.borrow();
            let pos1 = translate_quad_pos(QuadPos::NorthWest, self.plane, east_borrow.plane);
            let pos2 = translate_quad_pos(QuadPos::SouthWest, self.plane, east_borrow.plane);
            let q1 = east_borrow.get_child(pos1);
            let q2 = east_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags = PatchFlags::from(translate_quad_side(QuadSide::West, self.plane, east_borrow.plane));
            q1_borrow.patch_flags |= flags;
            q1_borrow.needs_normal_update = true;
            q1_borrow.needs_normal_merge = true;
            q2_borrow.patch_flags |= flags;
            q2_borrow.needs_normal_update = true;
            q2_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut east_borrow = direct_east.borrow_mut();
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            east_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(direct_east.clone());
        }

        if west_subdivided {
            let west_borrow = direct_west.borrow();
            let pos1 = translate_quad_pos(QuadPos::NorthEast, self.plane, west_borrow.plane);
            let pos2 = translate_quad_pos(QuadPos::SouthEast, self.plane, west_borrow.plane);
            let q1 = west_borrow.get_child(pos1);
            let q2 = west_borrow.get_child(pos2);
            let mut q1_borrow = q1.borrow_mut();
            let mut q2_borrow = q2.borrow_mut();
            let flags = PatchFlags::from(translate_quad_side(QuadSide::East, self.plane, west_borrow.plane));
            q1_borrow.patch_flags |= flags;
            q1_borrow.needs_normal_update = true;
            q1_borrow.needs_normal_merge = true;
            q2_borrow.patch_flags |= flags;
            q2_borrow.needs_normal_update = true;
            q2_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(q1.clone());
            sphere.queue_normal_update(q2.clone());
        } else {
            let mut west_borrow = direct_west.borrow_mut();
            // TODO: would this be strictly necessary if `merge_normals` updates
            // both quads?
            west_borrow.needs_normal_merge = true;
            sphere.queue_normal_update(direct_west.clone());
        }

        self.patch_flags = PATCH_FLAGS_NONE;
        self.needs_normal_update = true;
        self.needs_normal_merge = true;
        sphere.queue_normal_update(self.self_ptr.as_ref().unwrap().upgrade().unwrap());

        for c in self.children.as_ref().unwrap() {
            let mut c_borrow = c.borrow_mut();
            c_borrow.needs_normal_update = false;
            c_borrow.needs_normal_merge = false;
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
            self.collapse(sphere, scene);
        } else if self.is_subdivided() {
            for q in self.children.as_ref().unwrap() {
                q.borrow_mut().check_subdivision(sphere, scene);
            }
        }
    }

    fn update_normals(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        {
            let mesh = self.mesh.as_ref().unwrap();
            // FIXME: inefficient, should be fixed in up-coming quad pooling patch
            let indices = gen_indices(sphere.quad_mesh_size, self.patch_flags);
            *mesh.indices_mut(scene).unwrap() = indices;
        }
        self.calc_normals(scene);
    }

    fn update_edge_normals(&mut self, sphere: &QuadSphere, scene: &mut Scene) {
        self.merge_side_normals(sphere, scene, QuadSide::North);
        self.merge_side_normals(sphere, scene, QuadSide::South);
        self.merge_side_normals(sphere, scene, QuadSide::East);
        self.merge_side_normals(sphere, scene, QuadSide::West);
        self.merge_corner_normal(sphere, scene, QuadPos::NorthWest);
        self.merge_corner_normal(sphere, scene, QuadPos::NorthEast);
        self.merge_corner_normal(sphere, scene, QuadPos::SouthWest);
        self.merge_corner_normal(sphere, scene, QuadPos::SouthEast);
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
            mid_coord_pos: Vector3::new(0.0, 0.0, 0.0),
            patch_flags: PATCH_FLAGS_NONE,

            non_normalized: Vec::new(),

            needs_normal_update: false,
            needs_normal_merge: false,

            self_ptr: None,
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
    camera: Option<Rc<Camera>>,
    old_rot: Quaternion<f32>,
    prev_instant: std::time::Instant,
    ninety_deg: Quaternion<f32>,
    quad_mesh_size: u16,
    max_subdivision: u32,
    max_coord: u32,
    collapse_ranges: Vec<f32>,
    subdivide_ranges: Vec<f32>,
    centre_pos: Vector3<f32>,
    faces: Option<[Rc<RefCell<Quad>>; 6]>,
    normal_update_queue: RefCell<Vec<Rc<RefCell<Quad>>>>,
}

impl QuadSphere {
    fn init(&mut self, scene: &mut Scene, quad_mesh_size: u16, max_subdivision: u32) {
        assert!(quad_mesh_size > 1);
        let bits =  (quad_mesh_size as u32 - 1).leading_zeros();
        assert!(max_subdivision <= (bits - 1));

        self.quad_mesh_size = quad_mesh_size;
        self.max_subdivision = max_subdivision;
        self.max_coord = (1 << max_subdivision) * quad_mesh_size as u32;

        self.calc_ranges();

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
            let self_ptr = Rc::downgrade(&xp_quad);
            let mut xp_quad = xp_quad.borrow_mut();
            xp_quad.plane = Plane::XP;
            xp_quad.pos = QuadPos::None;
            xp_quad.self_ptr = Some(self_ptr);
            xp_quad.north = Some(Rc::downgrade(&yp_quad));
            xp_quad.south = Some(Rc::downgrade(&yn_quad));
            xp_quad.east = Some(Rc::downgrade(&zn_quad));
            xp_quad.west = Some(Rc::downgrade(&zp_quad));
            xp_quad.cur_subdivision = 0;
            xp_quad.base_coord = (0, 0);
            xp_quad.init(self, scene);
        }

        {
            let self_ptr = Rc::downgrade(&xn_quad);
            let mut xn_quad = xn_quad.borrow_mut();
            xn_quad.plane = Plane::XN;
            xn_quad.pos = QuadPos::None;
            xn_quad.self_ptr = Some(self_ptr);
            xn_quad.north = Some(Rc::downgrade(&yp_quad));
            xn_quad.south = Some(Rc::downgrade(&yn_quad));
            xn_quad.east = Some(Rc::downgrade(&zp_quad));
            xn_quad.west = Some(Rc::downgrade(&zn_quad));
            xn_quad.cur_subdivision = 0;
            xn_quad.base_coord = (0, 0);
            xn_quad.init(self, scene);
        }

        {
            let self_ptr = Rc::downgrade(&yp_quad);
            let mut yp_quad = yp_quad.borrow_mut();
            yp_quad.plane = Plane::YP;
            yp_quad.pos = QuadPos::None;
            yp_quad.self_ptr = Some(self_ptr);
            yp_quad.north = Some(Rc::downgrade(&zn_quad));
            yp_quad.south = Some(Rc::downgrade(&zp_quad));
            yp_quad.east = Some(Rc::downgrade(&xp_quad));
            yp_quad.west = Some(Rc::downgrade(&xn_quad));
            yp_quad.cur_subdivision = 0;
            yp_quad.base_coord = (0, 0);
            yp_quad.init(self, scene);
        }

        {
            let self_ptr = Rc::downgrade(&yn_quad);
            let mut yn_quad = yn_quad.borrow_mut();
            yn_quad.plane = Plane::YN;
            yn_quad.pos = QuadPos::None;
            yn_quad.self_ptr = Some(self_ptr);
            yn_quad.north = Some(Rc::downgrade(&zp_quad));
            yn_quad.south = Some(Rc::downgrade(&zn_quad));
            yn_quad.east = Some(Rc::downgrade(&xp_quad));
            yn_quad.west = Some(Rc::downgrade(&xn_quad));
            yn_quad.cur_subdivision = 0;
            yn_quad.base_coord = (0, 0);
            yn_quad.init(self, scene);
        }

        {
            let self_ptr = Rc::downgrade(&zp_quad);
            let mut zp_quad = zp_quad.borrow_mut();
            zp_quad.plane = Plane::ZP;
            zp_quad.pos = QuadPos::None;
            zp_quad.self_ptr = Some(self_ptr);
            zp_quad.north = Some(Rc::downgrade(&yp_quad));
            zp_quad.south = Some(Rc::downgrade(&yn_quad));
            zp_quad.east = Some(Rc::downgrade(&xp_quad));
            zp_quad.west = Some(Rc::downgrade(&xn_quad));
            zp_quad.cur_subdivision = 0;
            zp_quad.base_coord = (0, 0);
            zp_quad.init(self, scene);
        }

        {
            let self_ptr = Rc::downgrade(&zn_quad);
            let mut zn_quad = zn_quad.borrow_mut();
            zn_quad.plane = Plane::ZN;
            zn_quad.pos = QuadPos::None;
            zn_quad.self_ptr = Some(self_ptr);
            zn_quad.north = Some(Rc::downgrade(&yp_quad));
            zn_quad.south = Some(Rc::downgrade(&yn_quad));
            zn_quad.east = Some(Rc::downgrade(&xn_quad));
            zn_quad.west = Some(Rc::downgrade(&xp_quad));
            zn_quad.cur_subdivision = 0;
            zn_quad.base_coord = (0, 0);
            zn_quad.init(self, scene);
        }

        // TODO: reduce cloning
        let self_object = self.behaviour().object(scene).unwrap().clone();
        scene.set_object_parent(&xp_quad_obj, Some(&self_object));
        scene.set_object_parent(&xn_quad_obj, Some(&self_object));
        scene.set_object_parent(&yp_quad_obj, Some(&self_object));
        scene.set_object_parent(&yn_quad_obj, Some(&self_object));
        scene.set_object_parent(&zp_quad_obj, Some(&self_object));
        scene.set_object_parent(&zn_quad_obj, Some(&self_object));

        self.faces = Some([xp_quad, xn_quad, yp_quad, yn_quad, zp_quad, zn_quad]);
    }

    fn calc_ranges(&mut self)  {
        self.collapse_ranges = Vec::with_capacity(self.max_subdivision as usize + 1);
        self.subdivide_ranges = Vec::with_capacity(self.max_subdivision as usize + 1);
        for lvl in 0..(self.max_subdivision + 1) {
            let extra = (1.05 as f64).powf((self.max_subdivision - lvl) as f64);

            // Quad length changes when deformed into sphere
            let pi_div_4 = std::f64::consts::PI / 4.0;
            let quad_length = self.quad_length(lvl);
            let real_quad_length = 2.0 * (quad_length as f64 / self.max_coord as f64) * pi_div_4;

            let collapse_range = extra * 2.0 * 1.5 * real_quad_length;
            let subdivide_range = extra * 1.5 * real_quad_length;

            let r = 1.0;
            let collapse_cos_theta = f64::cos(f64::min(std::f64::consts::PI, collapse_range / r));
            let subdivide_cos_theta = f64::cos(f64::min(std::f64::consts::PI, subdivide_range / r));

            self.collapse_ranges.push(collapse_cos_theta as f32);
            self.subdivide_ranges.push(subdivide_cos_theta as f32);
        }
    }

    fn quad_length(&self, level: u32) -> u32 {
        (1 << (self.max_subdivision - level)) * (self.quad_mesh_size as u32)
    }

    fn centre_pos(&self) -> Vector3<f32> {
        self.centre_pos
    }

    /// Lookup the range required for us to try collapsing a quad for a given
    /// subdivision level. The returned value isn't a distance, but instead the
    /// cosine of the angle of the arc formed over that distance.
    fn collapse_range(&self, subdivision: u32) -> f32 {
        self.collapse_ranges[subdivision as usize]
    }

    /// Lookup the range required for us to try subdividing a quad for a given
    /// subdivision level. See `collapse_range()` for more details.
    fn subdivide_range(&self, subdivision: u32) -> f32 {
        self.subdivide_ranges[subdivision as usize]
    }

    fn queue_normal_update(&self, quad: Rc<RefCell<Quad>>) {
        self.normal_update_queue.borrow_mut().push(quad);
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

impl BehaviourMessages for QuadSphere {
    fn create(behaviour: Behaviour) -> QuadSphere {
        QuadSphere {
            behaviour: behaviour,
            camera: None,
            prev_instant: std::time::Instant::now(),
            old_rot: Quaternion::one(),
            ninety_deg: Quaternion::from(Euler { x: Deg(45.0), y: Deg(0.0), z: Deg(0.0) }),
            quad_mesh_size: 0,
            max_subdivision: 0,
            max_coord: 0,
            collapse_ranges: Vec::new(),
            subdivide_ranges: Vec::new(),
            centre_pos: Vector3::unit_z(),
            faces: None,
            normal_update_queue: RefCell::new(Vec::new()),
        }
    }

    fn start(&mut self, _scene: &mut Scene) {
        self.prev_instant = std::time::Instant::now();
    }

    fn update(&mut self, scene: &mut Scene) {
        let diff = self.prev_instant.elapsed();
        self.prev_instant = std::time::Instant::now();
        let secs = diff.as_secs() as f32 + diff.subsec_nanos() as f32 / 1000000000.0;

        let dps = 5.0;
        let change_rot = Quaternion::one().nlerp(self.ninety_deg, (dps / 45.0) * secs);

        // TODO: reduce cloning
        let rot = self.old_rot * change_rot;
        self.old_rot = rot;

        self.centre_pos = (rot.invert() * (Vector3::unit_z())).normalize();
        for i in 0..6 {
            let q = self.faces.as_ref().unwrap()[i].clone();
            q.borrow_mut().check_subdivision(self, scene);
        }

        for q in &*self.normal_update_queue.borrow() {
            let mut q = q.borrow_mut();
            if q.needs_normal_update {
                q.update_normals(self, scene);
            }
            q.needs_normal_update = false;
        }

        for q in &*self.normal_update_queue.borrow() {
            let mut q = q.borrow_mut();
            if q.needs_normal_merge {
                q.update_edge_normals(self, scene);
            }
            q.needs_normal_merge = false;
        }

        self.normal_update_queue.borrow_mut().clear();

        let cam_pos = 1.5f32 * self.centre_pos;
        let cam_rot = Quaternion::look_at(self.centre_pos, Vector3::unit_x()).invert();

        let camera = self.camera.as_ref().unwrap();
        camera.set_pos(scene, cam_pos).unwrap();
        camera.set_rot(scene, cam_rot).unwrap();
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

/// Maps a given vector and position on one plane to another. It allows other
/// code to deal with quads on differing planes as if they were on the same
/// plane, e.g.:
///
/// ```
/// +--------+--------+
/// |        |        |
/// |   YP   |   XP   |
/// |        |        |
/// +--------+--------+
/// ```
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
fn map_vec_pos<T: num::Signed + Copy>(vec: (T, T), pos: (T, T), max_coord: T, src_plane: Plane, dst_plane: Plane) -> ((T, T), (T, T)) {
    let (origin, dir_x, dir_y) = calc_plane_mapping::<T>(src_plane, dst_plane);
    let mapped_vec = (vec.0 * dir_x.0 + vec.1 * dir_y.0, vec.0 * dir_x.1 + vec.1 * dir_y.1);
    let origin = (origin.0 * max_coord, origin.1 * max_coord);
    let mapped_pos = (origin.0 + dir_x.0 * pos.0 + dir_y.0 * pos.1,
                      origin.1 + dir_x.1 * pos.0 + dir_y.1 * pos.1);
    (mapped_vec, mapped_pos)
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
    /// `side`. This doesn't update the corner normals.
    ///
    /// TODO: currently updates the edges but incorrectly, this is
    /// inconsequential.
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
        let quad_mesh_size = sphere.quad_mesh_size as i32;
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
                    QuadSide::West => ((0, 0), (max, 0))
                };

                let (q1_pos, q2_pos) = match side {
                    QuadSide::North => (QuadPos::SouthWest, QuadPos::SouthEast),
                    QuadSide::South => (QuadPos::NorthWest, QuadPos::NorthEast),
                    QuadSide::East => (QuadPos::SouthWest, QuadPos::NorthWest),
                    QuadSide::West => (QuadPos::SouthEast, QuadPos::NorthEast)
                };

                let q1_pos = translate_quad_pos(q1_pos, self.plane, direct_side.plane);
                let q2_pos = translate_quad_pos(q2_pos, self.plane, direct_side.plane);

                let q1 = direct_side.get_child(q1_pos);
                let q2 = direct_side.get_child(q2_pos);

                let (step_sx, step_sy) = (dir_sx, dir_sy);
                let (step_dx, step_dy) = (dir_sx * 2, dir_sy * 2);

                let ((step_dx, step_dy), (base_dx, base_dy)) = map_vec_pos(
                    (step_dx, step_dy),
                    (base_dx, base_dy),
                    quad_mesh_size,
                    self.plane, direct_side.plane);

                // TODO: don't touch corners
                self.merge_normals(sphere,
                                   scene,
                                   &mut *q1.borrow_mut(),
                                   (base_sx, base_sy),
                                   (step_sx, step_sy),
                                   (base_dx, base_dy),
                                   (step_dx, step_dy),
                                   quad_mesh_size / 2 + 1);

                let (base_sx2, base_sy2) = (base_sx + step_sx * half,
                                            base_sy + step_sy * half);

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
                    QuadSide::West => ((0, 0), (max, 0))
                };

                let (step_sx, step_sy) = (dir_sx, dir_sy);
                let (step_dx, step_dy) = (dir_sx, dir_sy);

                let ((step_dx, step_dy), (base_dx, base_dy)) = map_vec_pos(
                    (step_dx, step_dy),
                    (base_dx, base_dy),
                    quad_mesh_size,
                    self.plane, direct_side.plane);

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
            // FIXME: copy paste code
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
                QuadSide::West => ((0, 0), (max, y * half))
            };
            let (step_sx, step_sy) = (dir_sx * 2, dir_sy * 2);
            let (step_dx, step_dy) = (dir_sx, dir_sy);

            let ((step_dx, step_dy), (base_dx, base_dy)) = map_vec_pos(
                (step_dx, step_dy),
                (base_dx, base_dy),
                quad_mesh_size,
                self.plane, indirect_side.plane);

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
        let quad_mesh_size = sphere.quad_mesh_size as i32;
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

            let combined = self.non_normalized[svert_off] + other.non_normalized[dvert_off];
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

        let quad_mesh_size = sphere.quad_mesh_size as i32;
        let adj_size = quad_mesh_size + 1;
        let vert_off = |x, y| vert_off(x, y, adj_size as u16);
        let max = quad_mesh_size;
        let half = quad_mesh_size / 2;

        let (side1, side2) = corner.split();

        let (q1, q1_is_direct) = match self.get_direct_side(side1) {
            Some(q1) => (q1, true),
            None => (self.get_side(side1), false)
        };
        let (q2, q2_is_direct) = match self.get_direct_side(side2) {
            Some(q2) => (q2, true),
            None => (self.get_side(side2), false)
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
                _ => unreachable!()
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

        let (_, (q1_x, q1_y)) = map_vec_pos(
            (0, 0),
            (q1_x, q1_y),
            quad_mesh_size,
            self.plane, q1_plane);
        let q1_pos = translate_quad_pos(q1_pos, self.plane, q1_plane);

        let q2_x = match side2 {
            QuadSide::East => 0,
            QuadSide::West => max,
            _ => unreachable!(),
        };
        let q2_y = if q2_is_direct {
            match side1 {
                QuadSide::North => max,
                QuadSide::South => 0,
                _ => unreachable!()
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

        let (_, (q2_x, q2_y)) = map_vec_pos(
            (0, 0),
            (q2_x, q2_y),
            quad_mesh_size,
            self.plane, q2_plane);
        let q2_pos = translate_quad_pos(q2_pos, self.plane, q2_plane);

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

        if self.plane != q1_plane && self.plane != q2_plane && q1_plane != q2_plane || q1_x == half || q1_y == half || q2_x == half || q2_y == half {
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
                self.non_normalized[self_idx] +
                q1.non_normalized[q1_idx] +
                q2.non_normalized[q2_idx];
            let normalized = combined.normalize();

            self.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[self_idx] = normalized;
            q1.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[q1_idx] = normalized;
            q2.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[q2_idx] = normalized;
        } else {
            let side2 = translate_quad_side(side2, self.plane, q1_plane);

            let mut q3 = q1.borrow().get_direct_side(side2).unwrap_or_else(|| q1.borrow().get_side(side2));
            let q3_plane = q3.borrow().plane;
            let q3_pos = corner.opposite();
            let q3_pos = translate_quad_pos(q3_pos, self.plane, q3_plane);

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
                self.non_normalized[self_idx] +
                q1.non_normalized[q1_idx] +
                q2.non_normalized[q2_idx] +
                q3.non_normalized[q3_idx];
            let normalized = combined.normalize();

            self.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[self_idx] = normalized;
            q1.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[q1_idx] = normalized;
            q2.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[q2_idx] = normalized;
            q3.mesh.as_ref().unwrap().vnorm_mut(scene).unwrap()[q3_idx] = normalized;
        }
    }
}

fn main() {
    let display = glium::glutin::WindowBuilder::new()
        .with_multisampling(8)
        .with_depth_buffer(24)
        .build_glium().unwrap();

    let mut scene = Scene::new(display);
    let camera = scene.create_camera();
    camera.set_near_clip(&mut scene, 0.01).unwrap();
    camera.set_far_clip(&mut scene, 10.0).unwrap();

    let quad_sphere_obj = scene.create_object();
    let quad_sphere = scene.add_behaviour::<QuadSphere>(&quad_sphere_obj).unwrap();
    quad_sphere.borrow_mut().camera = Some(camera);
    quad_sphere.borrow_mut().init(&mut scene, 8, 5);
    quad_sphere_obj.set_world_pos(&mut scene, Vector3::new(0.0, 0.0, 0.0)).unwrap();
    //quad_sphere.borrow().object().set_world_rot(&mut scene, Quaternion::from(Euler { x: Deg(45.0), y: Deg(0.0), z: Deg(0.0) })).unwrap();

    loop {
        if !scene.do_frame() {
            break
        }
    }
}
