#![allow(dead_code)]

extern crate cgmath;
#[macro_use]
extern crate glium;

use cgmath::prelude::*;
use cgmath::{Deg, Euler, Matrix4, Quaternion, Vector3};
use glium::{Frame, IndexBuffer, Program, Surface, VertexBuffer};
use glium::index::PrimitiveType;
use glium::uniforms::UniformBuffer;
use glium::backend::glutin_backend::GlutinFacade;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::{Rc, Weak};
use std::slice::Iter;

#[derive(Copy, Clone)]
struct Vertex {
    vert_pos: [f32; 3],
}

implement_vertex!(Vertex, vert_pos);

/// Generate a `Vec` containing a `size + 1` by `size + 1` grid of vertices.
fn gen_vertices(size: u16) -> Vec<Vertex> {
    let adj_size = size + 1;
    let mut vertices = Vec::new();
    let diff = 2.0 / size as f32;
    for i in 0..adj_size {
        for j in 0..adj_size {
            vertices.push(Vertex { vert_pos: [-1.0 + i as f32 * diff, -1.0 + j as f32 * diff, 0.0] });
        }
    }
    vertices
}

/// Generate indices to fill the specified range on a grid of size `size`.
fn gen_indices_range(x1: u16, y1: u16, x2: u16, y2: u16, size: u16) -> Vec<u16> {
    // TODO: maybe debug_assert! this instead?
    let (x1, x2) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
    let (y1, y2) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
    let adj_size = size + 1;
    let mut indices = Vec::new();
    for i in x1..x2 {
        for j in y1..y2 {
            // *--------*
            // | 2    - |
            // |    -   |
            // |  -     |
            // |-    1  |
            // *--------*
            let t1a = i * adj_size + j;
            let t1b = (i + 1) * adj_size + j;
            let t1c = i * adj_size + (j + 1);

            let t2a = (i + 1) * adj_size + j;
            let t2b = (i + 1) * adj_size + (j + 1);
            let t2c = i * adj_size + (j + 1);

            indices.push(t1a);
            indices.push(t1b);
            indices.push(t1c);
            indices.push(t2a);
            indices.push(t2b);
            indices.push(t2c);
        }
    }
    indices
}

/// Utility function for computing the index of a vertex with a given stride.
fn vert_off(x: u16, y: u16, stride: u16) -> u16 {
    x * stride + y
}

/// Generate the indices for a quad of the given size and the specified
/// edges "patched".
fn gen_indices(size: u16, l: bool, r: bool, t: bool, b: bool) -> Vec<u16> {
    let adj_size = size + 1;
    let vert_off = |x, y| vert_off(x, y, adj_size);
    let mut indices = gen_indices_range(1, 1, size - 1, size - 1, size);
    if l {
        for i in 1..adj_size-1 {
            if (i % 2) == 1 {
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
                let ta = vert_off(0, i - 1);
                let tb = vert_off(1, i);
                let tc = vert_off(0, i + 1);

                indices.push(ta);
                indices.push(tb);
                indices.push(tc);
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
                let t1a = vert_off(1, i - 1);
                let t1b = vert_off(1, i);
                let t1c = vert_off(0, i);

                let t2a = vert_off(0, i);
                let t2b = vert_off(1, i);
                let t2c = vert_off(1, i + 1);

                indices.push(t1a);
                indices.push(t1b);
                indices.push(t1c);
                indices.push(t2a);
                indices.push(t2b);
                indices.push(t2c);
            }
        }
    } else {
        if b {
            let ta = vert_off(1, 1);
            let tb = vert_off(0, 1);
            let tc = vert_off(0, 0);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        } else {
            indices.extend(gen_indices_range(0, 0, 1, 1, size));
        }
        indices.extend(gen_indices_range(0, 1, 1, size - 1, size));
        if t {
            let ta = vert_off(0, size);
            let tb = vert_off(0, size - 1);
            let tc = vert_off(1, size - 1);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        } else {
            indices.extend(gen_indices_range(0, size - 1, 1, size, size));
        }
    }

    if t {
        for j in 1..adj_size-1 {
            if (j % 2) == 1 {
                let ta = vert_off(j - 1, size);
                let tb = vert_off(j, size - 1);
                let tc = vert_off(j + 1, size);

                indices.push(ta);
                indices.push(tb);
                indices.push(tc);
            } else {
                let t1a = vert_off(j - 1, size - 1);
                let t1b = vert_off(j, size - 1);
                let t1c = vert_off(j, size);

                let t2a = vert_off(j, size);
                let t2b = vert_off(j, size - 1);
                let t2c = vert_off(j + 1, size - 1);

                indices.push(t1a);
                indices.push(t1b);
                indices.push(t1c);
                indices.push(t2a);
                indices.push(t2b);
                indices.push(t2c);
            }
        }
    } else {
        if l {
            let ta = vert_off(1, size - 1);
            let tb = vert_off(1, size);
            let tc = vert_off(0, size);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        }
        indices.extend(gen_indices_range(1, size - 1, size - 1, size, size));
        if r {
            let ta = vert_off(size, size);
            let tb = vert_off(size - 1, size);
            let tc = vert_off(size - 1, size - 1);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        }
    }

    if b {
        for j in 1..adj_size-1 {
            if (j % 2) == 1 {
                //
                //          *
                //        -   -
                //      -       -
                //    -           -
                //  -               -
                // *-----------------*
                let ta = vert_off(j + 1, 0);
                let tb = vert_off(j, 1);
                let tc = vert_off(j - 1, 0);

                indices.push(ta);
                indices.push(tb);
                indices.push(tc);
            } else {
                // *--------*--------*
                //  -       |       -
                //    - 1   |  2  -
                //      -   |   -
                //        - | -
                //          *
                let t1a = vert_off(j, 0);
                let t1b = vert_off(j, 1);
                let t1c = vert_off(j - 1, 1);

                let t2a = vert_off(j + 1, 1);
                let t2b = vert_off(j, 1);
                let t2c = vert_off(j, 0);

                indices.push(t1a);
                indices.push(t1b);
                indices.push(t1c);
                indices.push(t2a);
                indices.push(t2b);
                indices.push(t2c);
            }
        }
    } else {
        if l {
            let ta = vert_off(0, 0);
            let tb = vert_off(1, 0);
            let tc = vert_off(1, 1);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        }
        indices.extend(gen_indices_range(1, 0, size - 1, 1, size));
        if r {
            let ta = vert_off(size - 1, 1);
            let tb = vert_off(size - 1, 0);
            let tc = vert_off(size, 0);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        }
    }

    if r {
        for i in 1..adj_size-1 {
            if (i % 2) == 1 {
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
                let ta = vert_off(adj_size - 1, i + 1);
                let tb = vert_off(adj_size - 2, i);
                let tc = vert_off(adj_size - 1, i - 1);

                indices.push(ta);
                indices.push(tb);
                indices.push(tc);
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
                let t1a = vert_off(adj_size - 2, i);
                let t1b = vert_off(adj_size - 2, i - 1);
                let t1c = vert_off(adj_size - 1, i);

                let t2a = vert_off(adj_size - 2, i + 1);
                let t2b = vert_off(adj_size - 2, i);
                let t2c = vert_off(adj_size - 1, i);

                indices.push(t1a);
                indices.push(t1b);
                indices.push(t1c);
                indices.push(t2a);
                indices.push(t2b);
                indices.push(t2c);
            }
        }
    } else {
        if b {
            let ta = vert_off(size, 0);
            let tb = vert_off(size, 1);
            let tc = vert_off(size - 1, 1);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        } else {
            indices.extend(gen_indices_range(size - 1, 0, size, 1, size));
        }
        indices.extend(gen_indices_range(size - 1, 1, size, size - 1, size));
        if t {
            let ta = vert_off(size - 1, size - 1);
            let tb = vert_off(size, size - 1);
            let tc = vert_off(size, size);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        } else {
            indices.extend(gen_indices_range(size - 1, size - 1, size, size, size));
        }
    }
    indices
}

pub struct ObjectData {
    rot: Quaternion<f32>,
    pos: Vector3<f32>,
    children: Vec<Rc<RefCell<Object>>>,
    //mesh: Option<Mesh>
}

pub struct ChildrenIter<'a> {
    iter: Iter<'a, Rc<RefCell<Object>>>,
}

impl<'a> Iterator for ChildrenIter<'a> {
    type Item = std::cell::Ref<'a, Object>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|child| child.borrow())
    }
}

pub trait Object {
    //fn on_start(&self);
    fn on_frame(&self, scene: &Scene, display: &GlutinFacade,
                target: &mut Frame, draw_params: &glium::DrawParameters,
                camera_ubo: &UniformBuffer<CameraBufData>);
    fn object_data(&self) -> &ObjectData;
    fn children(&self) -> ChildrenIter;
}

struct Quad {
    data: ObjectData,
    vertex_buf: VertexBuffer<Vertex>,
    indices_buf: IndexBuffer<u16>,
    program: Rc<Program>,
    parent: Option<Weak<RefCell<Quad>>>,
    //// The child quads if this quad has been subdivided
    // TODO: this requires Copy, would rather not implement Copy
    //children: [Option<Rc<RefCell<Quad>>>; 4],
}

// TODO: impl Object for Quad
impl Object for Quad {
    fn on_frame(&self, scene: &Scene, display: &GlutinFacade,
                target: &mut Frame, draw_params: &glium::DrawParameters,
                camera_ubo: &UniformBuffer<CameraBufData>) {
        // TODO: only recalc when necessary
        let tmp_matrix =
            Matrix4::from(self.data.rot) *
            Matrix4::from_translation(self.data.pos);
        let tmp_matrix: [[f32; 4]; 4] = tmp_matrix.clone().into();
        target.draw(&self.vertex_buf, &self.indices_buf, &self.program,
                    &uniform! {
                        CamBuffer: camera_ubo,
                        obj_matrix: tmp_matrix,
                    },
                    draw_params).unwrap();
    }

    fn object_data(&self) -> &ObjectData {
        &self.data
    }

    fn children(&self) -> ChildrenIter {
        ChildrenIter { iter: self.data.children.iter() }
    }
}

impl Quad {
    fn new(scene: &Scene, display: &GlutinFacade, pos: Vector3<f32>, rot: Quaternion<f32>) -> Quad {
        let vertices = gen_vertices(16);
        let indices = gen_indices(16, false, false, false, false);
        Quad {
            data: ObjectData {
                pos: pos,
                rot: rot,
                children: Vec::new()
            },
            vertex_buf: VertexBuffer::new(display, &vertices).unwrap(),
            indices_buf: IndexBuffer::new(display, PrimitiveType::TrianglesList, &indices).unwrap(),
            program: scene.program("default").clone(),
            parent: None
            //children: [None; 4]
        }
    }
}

struct QuadSphere {
    /// The six faces of the cube
    faces: [Option<Rc<RefCell<Quad>>>; 6],
}

// TODO: impl Object for QuadSphere

struct QuadSphereManager {
    quad_stack: Vec<Rc<RefCell<Quad>>>,
}

// TODO: impl Object for QuadSphereManager

#[derive(Copy, Clone)]
pub struct CameraBufData {
    cam_matrix: [[f32; 4]; 4],
}

//implement_buffer_content!(CameraBufData);
implement_uniform_block!(CameraBufData, cam_matrix);

struct Camera {
    rot: Quaternion<f32>,
    pos: Vector3<f32>,
    fovy: Deg<f32>,
    aspect: f32,
    near_clip: f32,
    far_clip: f32,
    buf: UniformBuffer<CameraBufData>
}

impl Camera {
    fn new(display: &GlutinFacade) -> Camera {
        Camera {
            rot: Quaternion::from(Euler {
                x: Deg(0f32),
                y: Deg(0f32),
                z: Deg(0f32)
            }),
            pos: Vector3::new(0f32, 0f32, 0f32),
            fovy: Deg(90f32),
            aspect: 1f32,
            near_clip: 1f32,
            far_clip: 1000f32,
            buf: UniformBuffer::new(display, CameraBufData {
                cam_matrix: Default::default(),
            }).unwrap()
        }
    }

    fn update_matrix(&mut self) {
        let cam_perspective = cgmath::perspective(self.fovy, self.aspect, self.near_clip, self.far_clip);
        let cam_matrix =
            cam_perspective *
            Matrix4::from(self.rot.invert()) *
            Matrix4::from_translation(-self.pos);
        self.buf.map().cam_matrix = cam_matrix.clone().into();
    }
}

// TODO: impl Object for Camera

#[derive(Default)]
pub struct Scene {
    cameras: Vec<Camera>,
    children: Vec<Rc<RefCell<Object>>>,
    programs: HashMap<String, Rc<Program>>,
}



impl Scene {
    fn new() -> Scene {
        Default::default()
    }

    fn add_camera(&mut self, cam: Camera) {
        self.cameras.push(cam);
    }

    fn add_object<T: Object + 'static>(&mut self, obj: T) {
        self.children.push(Rc::new(RefCell::new(obj)));
    }

    fn add_program(&mut self, prog_name: &str, prog: Program) {
        self.programs.insert(prog_name.to_owned(), Rc::new(prog));
    }

    fn camera_ubo(&self, cam_idx: usize) -> &UniformBuffer<CameraBufData> {
        &self.cameras[cam_idx].buf
    }

    fn program(&self, prog_name: &str) -> &Rc<Program> {
        &self.programs[prog_name]
    }

    fn on_frame(&mut self, display: &GlutinFacade, target: &mut Frame) {
        target.clear_color_and_depth((0.8, 0.8, 0.8, 1.0), 1.0);

        let draw_params = glium::DrawParameters {
            backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
            polygon_mode: glium::draw_parameters::PolygonMode::Line,
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };

        let (width, height) = target.get_dimensions();
        let aspect = width as f32 / height as f32;
        for cam in &mut self.cameras {
            if cam.aspect != aspect {
                cam.aspect = aspect;
                cam.update_matrix();
            }
        }
        // TODO: support multiple cameras
        for c in &self.children {
            c.borrow().on_frame(&self, display, target, &draw_params, self.camera_ubo(0));
        }
    }
}

fn main() {
    use glium::DisplayBuild;
    let display = glium::glutin::WindowBuilder::new()
        .with_multisampling(8)
        .with_depth_buffer(24)
        .build_glium().unwrap();

    let mut scene = Scene::new();
    let default_vs = include_str!("default_vs.glsl");
    let default_fs = include_str!("default_fs.glsl");
    let default_prog = Program::from_source(&display, default_vs, default_fs, None).unwrap();
    scene.add_program("default", default_prog);
    let mut camera = Camera::new(&display);
    camera.near_clip = 0.1f32;
    camera.far_clip = 1000f32;
    scene.add_camera(camera);

    let q1 = Quad::new(&scene, &display, Vector3::new(0f32, 0f32, -1.5), Quaternion::from(Euler {
        x: Deg(0f32),
        y: Deg(0f32),
        z: Deg(0f32)
    }));
    scene.add_object(q1);

    loop {
        let mut target = display.draw();
        scene.on_frame(&display, &mut target);
        target.finish().unwrap();

        for ev in display.poll_events() {
            match ev {
                glium::glutin::Event::Closed => return,
                _ => std::thread::sleep_ms(1)
            }
        }
    }
}
