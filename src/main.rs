#[macro_use]
extern crate bitflags;
extern crate cgmath;
extern crate glium;
extern crate num;
extern crate planetgen_engine;

use std::rc::Rc;

use cgmath::{Deg, Euler, Quaternion, Vector3};

use glium::DisplayBuild;

use num::One;

use planetgen_engine::{Behaviour, Material, Mesh, Object, Scene, Shader, Vertex};

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

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy)]
struct VertCoord(Plane, u32, u32);

struct Quad {
    object: Object,
    plane: Plane,
    mesh: Option<Rc<Mesh>>,
    shader: Option<Rc<Shader>>,
    material: Option<Rc<Material>>,
}

impl Behaviour for Quad {
    fn create(object: Object) -> Quad {
        Quad {
            object: object,
            plane: Plane::XP,
            mesh: None,
            shader: None,
            material: None,
        }
    }

    fn start(&mut self, scene: &mut Scene) {
        let quad_size: u16 = 16;
        let base_coord = (0, 0);
        let max_subdivision: u32 = 10;
        let cur_subdivision: u32 = 0;
        let max_coord: u32 = (1 << max_subdivision) * quad_size as u32;
        let vert_step = 1 << (max_subdivision - cur_subdivision);
        let adj_size = quad_size + 1;
        let mut vertices = gen_vertices(quad_size);
        let indices = gen_indices(quad_size, PATCH_SIDE_NONE);
        println!("Plane = {:?}", self.plane);

        let vert_off = |x, y| vert_off(x, y, adj_size);
        for x in 0..adj_size {
            for y in 0..adj_size {
                let vert_coord = VertCoord(self.plane,
                                           base_coord.0 + x as u32 * vert_step,
                                           base_coord.1 + y as u32 * vert_step);
                use cgmath::InnerSpace;
                let vert_pos = map_vertcoord(vert_coord, max_coord).normalize();
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
    }

    fn update(&mut self, _scene: &mut Scene) {
    }

    fn destroy(&mut self, _scene: &mut Scene) {
    }

    fn object(&self) -> &Object {
        &self.object
    }

    fn mesh(&self) -> Option<&Mesh> {
        self.mesh.as_ref().map(|mesh| &**mesh)
    }

    fn material(&self) -> Option<&Material> {
        self.material.as_ref().map(|material| &**material)
    }
}

struct QuadSphere {
    object: Object,
    prev_instant: std::time::Instant,
    ninety_deg: Quaternion<f32>,
}

impl Behaviour for QuadSphere {
    fn create(object: Object) -> QuadSphere {
        QuadSphere {
            object: object,
            prev_instant: std::time::Instant::now(),
            ninety_deg: Quaternion::from(Euler { x: Deg(0.0), y: Deg(45.0), z: Deg(0.0) }),
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
        let change_rot = Quaternion::one().nlerp(self.ninety_deg, (dps / 45.0) * secs);//self.ninety_deg * ((dps / 45.0) * 10 * secs);

        let rot = self.object.local_rot(scene).unwrap();
        let rot = rot * change_rot;
        self.object.set_local_rot(scene, rot).unwrap();
    }

    fn destroy(&mut self, _scene: &mut Scene) {
    }

    fn object(&self) -> &Object {
        &self.object
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

    let xp_quad = scene.create_object::<Quad>();
    xp_quad.borrow_mut().plane = Plane::XP;
    let xn_quad = scene.create_object::<Quad>();
    xn_quad.borrow_mut().plane = Plane::XN;
    let yp_quad = scene.create_object::<Quad>();
    yp_quad.borrow_mut().plane = Plane::YP;
    let yn_quad = scene.create_object::<Quad>();
    yn_quad.borrow_mut().plane = Plane::YN;
    let zp_quad = scene.create_object::<Quad>();
    zp_quad.borrow_mut().plane = Plane::ZP;
    let zn_quad = scene.create_object::<Quad>();
    zn_quad.borrow_mut().plane = Plane::ZN;

    let quad_sphere = scene.create_object::<QuadSphere>();
    {
        let quad_sphere_borrow = quad_sphere.borrow();
        let quad_sphere_object = quad_sphere_borrow.object();
        scene.set_object_parent(xp_quad.borrow().object(), Some(quad_sphere_object));
        scene.set_object_parent(xn_quad.borrow().object(), Some(quad_sphere_object));
        scene.set_object_parent(yp_quad.borrow().object(), Some(quad_sphere_object));
        scene.set_object_parent(yn_quad.borrow().object(), Some(quad_sphere_object));
        scene.set_object_parent(zp_quad.borrow().object(), Some(quad_sphere_object));
        scene.set_object_parent(zn_quad.borrow().object(), Some(quad_sphere_object));
    }

    quad_sphere.borrow().object().set_world_pos(&mut scene, Vector3::new(0.0, 0.0, -2.5)).unwrap();
    //quad_sphere.borrow().object().set_world_rot(&mut scene, Quaternion::from(Euler { x: Deg(45.0), y: Deg(0.0), z: Deg(0.0) })).unwrap();

    loop {
        if !scene.do_frame() {
            break
        }
    }
}
