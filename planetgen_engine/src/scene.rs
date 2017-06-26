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

use alloc::{AllocRange, SimpleAllocator};
use cgmath;
use cgmath::{Deg, Euler, Matrix4, Quaternion, Rotation, Matrix, Vector3};
use error::{Error, Result};
use gl;
use gl::types::*;
use gl_util::Program;
use num::{Zero, One};
use sdl2::{EventPump, Sdl};
use sdl2::event::Event;
use sdl2::keyboard::KeyboardState;
use sdl2::mouse::MouseState;
use sdl2::video::{GLContext, GLProfile, Window};
use std;
use std::any::{Any, TypeId};
use std::cell::{Cell, UnsafeCell};
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::ffi::CStr;
use std::ptr;
use std::rc::Rc;
use std::time::Instant;
use obj_manager::{GenericHandle, Container, ObjectManager};
use traits::Component;

pub use obj_manager::Handle;

fn post_add<T: Copy + std::ops::Add<Output=T>>(a: &mut T, b: T) -> T {
    let c = *a;
    *a = *a + b;
    c
}

fn calc_aabb(vertices: &[Vector3<f32>]) -> ([f32; 2], [f32; 2], [f32; 2]) {
    if vertices.len() == 0 {
        return ([0.0, 0.0], [0.0, 0.0], [0.0, 0.0]);
    }

    let (mut min_x, mut min_y, mut min_z) = (std::f32::MAX, std::f32::MAX, std::f32::MAX);
    let (mut max_x, mut max_y, mut max_z) = (std::f32::MIN, std::f32::MIN, std::f32::MIN);
    for v in vertices {
        min_x = f32::min(min_x, v.x);
        min_y = f32::min(min_y, v.y);
        min_z = f32::min(min_z, v.z);
        max_x = f32::max(max_x, v.x);
        max_y = f32::max(max_y, v.y);
        max_z = f32::max(max_z, v.z);
    }

    ([min_x, max_x], [min_y, max_y], [min_z, max_z])

}

pub fn intersects_frustum(mvp: Matrix4<f32>, aabb_x: [f32; 2], aabb_y: [f32; 2], aabb_z: [f32; 2]) -> bool {
    // ZP
    let zp_plane = -mvp.z + mvp.w;
    let x = aabb_x[(zp_plane.x > 0.0) as usize];
    let y = aabb_y[(zp_plane.y > 0.0) as usize];
    let z = aabb_z[(zp_plane.z > 0.0) as usize];
    if zp_plane.x * x + zp_plane.y * y + zp_plane.z * z + zp_plane.w < 0.0 {
        return false;
    }

    // XP
    let xp_plane = -mvp.x + mvp.w;
    let x = aabb_x[(xp_plane.x > 0.0) as usize];
    let y = aabb_y[(xp_plane.y > 0.0) as usize];
    let z = aabb_z[(xp_plane.z > 0.0) as usize];
    if xp_plane.x * x + xp_plane.y * y + xp_plane.z * z + xp_plane.w < 0.0 {
        return false;
    }

    // XN
    let xn_plane = mvp.x + mvp.w;
    let x = aabb_x[(xn_plane.x > 0.0) as usize];
    let y = aabb_y[(xn_plane.y > 0.0) as usize];
    let z = aabb_z[(xn_plane.z > 0.0) as usize];
    if xn_plane.x * x + xn_plane.y * y + xn_plane.z * z + xn_plane.w < 0.0 {
        return false;
    }

    // YP
    let yp_plane = -mvp.y + mvp.w;
    let x = aabb_x[(yp_plane.x > 0.0) as usize];
    let y = aabb_y[(yp_plane.y > 0.0) as usize];
    let z = aabb_z[(yp_plane.z > 0.0) as usize];
    if yp_plane.x * x + yp_plane.y * y + yp_plane.z * z + yp_plane.w < 0.0 {
        return false;
    }

    // YN
    let yn_plane = mvp.y + mvp.w;
    let x = aabb_x[(yn_plane.x > 0.0) as usize];
    let y = aabb_y[(yn_plane.y > 0.0) as usize];
    let z = aabb_z[(yn_plane.z > 0.0) as usize];
    if yn_plane.x * x + yn_plane.y * y + yn_plane.z * z + yn_plane.w < 0.0 {
        return false;
    }

    // ZN
    let zn_plane = mvp.z + mvp.w;
    let x = aabb_x[(zn_plane.x > 0.0) as usize];
    let y = aabb_y[(zn_plane.y > 0.0) as usize];
    let z = aabb_z[(zn_plane.z > 0.0) as usize];
    if zn_plane.x * x + zn_plane.y * y + zn_plane.z * z + zn_plane.w < 0.0 {
        return false;
    }

    true
}

struct CameraData {
    /// Reference to the object we are a component of.
    parent: Handle<Object>,
    /// True when this camera should be used for rendering.
    enabled: bool,
    /// True when the camera has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    /// An integer describing the order in which this camera is to be
    /// renderered, higher values mean later.
    order: i32,
    /// Bitmask containing the layers this camera renders
    layers: i32,
    /// The vertical FOV for this camera.
    fovy: Deg<f32>,
    /// The aspect ratio for this camera.
    aspect: f32,
    /// Near clip plane distance.
    near_clip: f32,
    /// Far clip plane distance.
    far_clip: f32,
    /// The position of the camera in world coordinates.
    pos: Vector3<f32>,
    /// The Projection-View matrix for this camera.
    pv_matrix: Matrix4<f32>,
    /// A vector containing the draw commands to be executed using this camera.
    draw_cmds: Vec<DrawInfo>,
}

struct CameraContainer {
    data: Vec<CameraData>,
}

impl Container for CameraContainer {
    type Item = CameraData;
    type HandleType = Camera;

    fn push(&mut self, value: Self::Item) {
        self.data.push(value);
    }

    fn swap_remove(&mut self, idx: usize) {
        self.data.swap_remove(idx);
    }
}

impl CameraContainer {
    fn new() -> CameraContainer {
        CameraContainer {
            data: Vec::new(),
        }
    }
}

/// A handle to a camera object for a scene.
#[derive(Copy, Clone)]
pub struct Camera;

impl Handle<Camera> {
    pub fn set_fovy(&self, scene: &mut Scene, fovy: f32) -> Result<()> {
        let idx = scene.camera_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &mut scene.camera_data.c.data[i];
                data.fovy = Deg(fovy);
            })
    }

    pub fn set_near_clip(&self, scene: &mut Scene, near_clip: f32) -> Result<()> {
        let idx = scene.camera_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &mut scene.camera_data.c.data[i];
                data.near_clip = near_clip;
            })
    }

    pub fn set_far_clip(&self, scene: &mut Scene, far_clip: f32) -> Result<()> {
        let idx = scene.camera_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &mut scene.camera_data.c.data[i];
                data.far_clip = far_clip;
            })
    }

    pub fn set_order(&self, scene: &mut Scene, order: i32) -> Result<()> {
        let idx = scene.camera_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &mut scene.camera_data.c.data[i];
                data.order = order;
            })
    }

    pub fn set_layers(&self, scene: &mut Scene, layers: i32) -> Result<()> {
        let idx = scene.camera_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &mut scene.camera_data.c.data[i];
                data.layers = layers;
            })
    }

    pub fn order(&self, scene: &Scene) -> Result<i32> {
        let idx = scene.camera_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &scene.camera_data.c.data[i];
                data.order
            })
    }

    pub fn layers(&self, scene: &Scene) -> Result<i32> {
        let idx = scene.camera_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &scene.camera_data.c.data[i];
                data.layers
            })
    }

    pub fn is_valid(&self, scene: &Scene) -> bool {
        scene.camera_data.is_handle_valid(*self)
    }
}

impl Component for Camera {
    fn init(scene: &mut Scene, object: Handle<Object>) -> Result<Handle<Camera>> {
        scene.create_camera(object)
    }

    fn marked(component: Handle<Camera>, scene: &Scene) -> Result<bool> {
        let idx = scene.camera_data.data_idx_checked(component);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                scene.camera_data.c.data[i].marked
            })
    }

    fn destroy(component: Handle<Camera>, scene: &mut Scene) {
        scene.destroy_camera(component);
    }
}

// TODO: this structure is getting quite large, consider splitting it up?
struct MeshData {
    /// Reference to the mesh object.
    object: Rc<Mesh>,
    /// True when the mesh has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    vpos_vec: Vec<Vector3<f32>>,
    vnorm_vec: Vec<Vector3<f32>>,
    vcolour_vec: Vec<Vector3<f32>>,
    indices_vec: Vec<u16>,
    /// True if vertex buffer data needs to be updated.
    vertex_buf_dirty: bool,
    /// A tuple containing a vertex buffer index and an allocation range
    /// indicating where the vertex data for this mesh is. `None` if not
    /// allocated yet.
    vertex_buf_alloc: Option<(usize, AllocRange)>,
    // TODO: AllocRange stores this as well, use that instead
    vertex_buf_capacity: usize,
    /// True if index buffer data needs to be updated.
    index_buf_dirty: bool,
    /// A tuple containing a index buffer index and an allocation range
    /// indicating where the index data for this mesh is. `None` if not
    /// allocated yet.
    index_buf_alloc: Option<(usize, AllocRange)>,
    index_buf_capacity: usize,
    aabb_dirty: bool,
    aabb_x: [f32; 2],
    aabb_y: [f32; 2],
    aabb_z: [f32; 2],
}

impl MeshData {
    fn vertex_buf_len(&self) -> usize {
        *[self.vpos_vec.len(), self.vnorm_vec.len(), self.vcolour_vec.len()].iter().max().unwrap()
    }
}

pub struct Mesh {
    idx: Cell<Option<usize>>
}

impl Mesh {
    pub fn vpos<'a>(&self, scene: &'a Scene) -> Result<&'a Vec<Vector3<f32>>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                &scene.mesh_data[i].vpos_vec
            })
    }

    pub fn vpos_mut<'a>(&self, scene: &'a mut Scene) -> Result<&'a mut Vec<Vector3<f32>>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(move |i| {
                let data = &mut scene.mesh_data[i];
                data.vertex_buf_dirty = true;
                data.aabb_dirty = true;
                &mut data.vpos_vec
            })
    }

    pub fn vnorm<'a>(&self, scene: &'a Scene) -> Result<&'a Vec<Vector3<f32>>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                &scene.mesh_data[i].vnorm_vec
            })
    }

    pub fn vnorm_mut<'a>(&self, scene: &'a mut Scene) -> Result<&'a mut Vec<Vector3<f32>>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(move |i| {
                let data = &mut scene.mesh_data[i];
                data.vertex_buf_dirty = true;
                &mut data.vnorm_vec
            })
    }

    pub fn vcolour<'a>(&self, scene: &'a Scene) -> Result<&'a Vec<Vector3<f32>>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                &scene.mesh_data[i].vcolour_vec
            })
    }

    pub fn vcolour_mut<'a>(&self, scene: &'a mut Scene) -> Result<&'a mut Vec<Vector3<f32>>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(move |i| {
                let data = &mut scene.mesh_data[i];
                data.vertex_buf_dirty = true;
                &mut data.vcolour_vec
            })
    }

    pub fn indices<'a>(&self, scene: &'a Scene) -> Result<&'a Vec<u16>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                &scene.mesh_data[i].indices_vec
            })
    }

    pub fn indices_mut<'a>(&self, scene: &'a mut Scene) -> Result<&'a mut Vec<u16>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(move |i| {
                let data = &mut scene.mesh_data[i];
                data.index_buf_dirty = true;
                &mut data.indices_vec
            })
    }
}

struct CubemapData {
    /// True when the cubemap has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    texture_id: GLuint,
}

struct CubemapContainer {
    data: Vec<CubemapData>,
}

impl Container for CubemapContainer {
    type Item = CubemapData;
    type HandleType = Cubemap;

    fn push(&mut self, value: Self::Item) {
        self.data.push(value);
    }

    fn swap_remove(&mut self, idx: usize) {
        self.data.swap_remove(idx);
    }
}

impl CubemapContainer {
    fn new() -> CubemapContainer {
        CubemapContainer {
            data: Vec::new(),
        }
    }
}

/// A handle to a cubemap object for a scene.
#[derive(Copy, Clone)]
pub struct Cubemap;

struct DrawInfo {
    shader_idx: usize,
    material_idx: usize,
    vertex_buf_idx: usize,
    index_buf_idx: usize,
    trans_idx: usize,
    mesh_idx: usize,
}

struct ShaderData {
    /// True when the shader has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    program: Program,
    obj_matrix_uniform: GLint,
    cam_pos_uniform: GLint,
    cam_matrix_uniform: GLint,
    colour_uniform: GLint,
}

struct ShaderContainer {
    data: Vec<ShaderData>,
}

impl Container for ShaderContainer {
    type Item = ShaderData;
    type HandleType = Shader;

    fn push(&mut self, value: Self::Item) {
        self.data.push(value);
    }

    fn swap_remove(&mut self, idx: usize) {
        self.data.swap_remove(idx);
    }
}

impl ShaderContainer {
    fn new() -> ShaderContainer {
        ShaderContainer {
            data: Vec::new(),
        }
    }
}

/// A handle to a shader object for a scene.
#[derive(Copy, Clone)]
pub struct Shader;

impl Handle<Shader> {
    pub fn is_valid(&self, scene: &Scene) -> bool {
        scene.shader_data.is_handle_valid(*self)
    }
}

pub enum UniformValue {
    Mat2([[f32; 2]; 2]),
    Mat3([[f32; 3]; 3]),
    Mat4([[f32; 4]; 4]),
    Float(f32),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    Int(i32),
    IntVec2([i32; 2]),
    IntVec3([i32; 3]),
    IntVec4([i32; 4]),
    UInt(u32),
    UIntVec2([u32; 2]),
    UIntVec3([u32; 3]),
    UIntVec4([u32; 4]),
    Cubemap(Handle<Cubemap>),
    //Bool(bool),
    //BoolVec2([bool; 2]),
    //BoolVec3([bool; 3]),
    //BoolVec4([bool; 4]),
    //Double(f64),
    //DoubleVec2([f64; 2]),
    //DoubleVec3([f64; 3]),
    //DoubleVec4([f64; 4]),
    //DoubleMat2([[f64; 2]; 2]),
    //DoubleMat3([[f64; 3]; 3]),
    //DoubleMat4([[f64; 4]; 4]),
    //Int64(i64),
    //Int64Vec2([i64; 2]),
    //Int64Vec3([i64; 3]),
    //Int64Vec4([i64; 4]),
    //UInt64(u64),
    //UInt64Vec2([u64; 2]),
    //UInt64Vec3([u64; 3]),
    //UInt64Vec4([u64; 4])
}

struct MaterialData {
    /// True when the material has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    uniforms: HashMap<String, usize>,
    uniform_values: Vec<(GLint, Option<UniformValue>)>,
    shader: Handle<Shader>,
    colour: (f32, f32, f32)
}

struct MaterialContainer {
    data: Vec<MaterialData>,
}

impl Container for MaterialContainer {
    type Item = MaterialData;
    type HandleType = Material;

    fn push(&mut self, value: Self::Item) {
        self.data.push(value);
    }

    fn swap_remove(&mut self, idx: usize) {
        self.data.swap_remove(idx);
    }
}

impl MaterialContainer {
    fn new() -> MaterialContainer {
        MaterialContainer {
            data: Vec::new(),
        }
    }
}

/// A handle to a camera object for a scene.
#[derive(Copy, Clone)]
pub struct Material;

impl Handle<Material> {
    pub fn set_colour(&self, scene: &mut Scene, colour: (f32, f32, f32)) -> Result<()> {
        let idx = scene.material_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| { scene.material_data.c.data[i].colour = colour; })
    }

    pub fn colour(&self, scene: &Scene) -> Result<(f32, f32, f32)> {
        let idx = scene.material_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| { scene.material_data.c.data[i].colour })
    }

    pub fn set_uniform(&self, scene: &mut Scene, name: &str, v: UniformValue) -> Result<()> {
        let material_data = try!(
            scene.material_data.data_idx_checked(*self)
                .or(Err(Error::ObjectDestroyed))
                .map(|i| &mut scene.material_data.c.data[i]));

        match material_data.uniforms.get(name) {
            Some(&idx) => {
                material_data.uniform_values[idx].1 = Some(v);
                Ok(())
            },
            None => Err(Error::BadUniformName)
        }
    }

    pub fn is_valid(&self, scene: &Scene) -> bool {
        scene.material_data.is_handle_valid(*self)
    }
}

struct MeshRendererData {
    /// Reference to the object we are a component of.
    parent: Handle<Object>,
    /// True when the mesh renderer has been marked for destruction at the end
    /// of the frame.
    marked: bool,
    enabled: bool,
    /// Bitmask containing the layers this mesh is rendered on.
    layers: i32,
    mesh: Option<Rc<Mesh>>,
    material: Option<Handle<Material>>,
}

struct MeshRendererContainer {
    data: Vec<MeshRendererData>,
}

impl Container for MeshRendererContainer {
    type Item = MeshRendererData;
    type HandleType = MeshRenderer;

    fn push(&mut self, value: Self::Item) {
        self.data.push(value);
    }

    fn swap_remove(&mut self, idx: usize) {
        self.data.swap_remove(idx);
    }
}

impl MeshRendererContainer {
    fn new() -> MeshRendererContainer {
        MeshRendererContainer {
            data: Vec::new(),
        }
    }
}

/// A handle to a mesh renderer object for a scene.
#[derive(Copy, Clone)]
pub struct MeshRenderer;

impl Handle<MeshRenderer> {
    pub fn set_enabled(&self, scene: &mut Scene, enabled: bool) -> Result<()> {
        let idx = scene.mrenderer_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| scene.mrenderer_data.c.data[i].enabled = enabled)
    }

    pub fn set_mesh(&self, scene: &mut Scene, mesh: Option<Rc<Mesh>>) -> Result<()> {
        let idx = scene.mrenderer_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| scene.mrenderer_data.c.data[i].mesh = mesh)
    }

    pub fn set_material(&self, scene: &mut Scene, material: Option<Handle<Material>>) -> Result<()> {
        let idx = scene.mrenderer_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| scene.mrenderer_data.c.data[i].material = material)
    }

    pub fn set_layers(&self, scene: &mut Scene, layers: i32) -> Result<()> {
        let idx = scene.mrenderer_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &mut scene.mrenderer_data.c.data[i];
                data.layers = layers;
            })
    }

    pub fn enabled(&self, scene: &mut Scene) -> Result<bool> {
        let idx = scene.mrenderer_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| scene.mrenderer_data.c.data[i].enabled)
    }

    pub fn mesh<'a>(&self, scene: &'a mut Scene) -> Result<Option<&'a Rc<Mesh>>> {
        let idx = scene.mrenderer_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(move |i| scene.mrenderer_data.c.data[i].mesh.as_ref())
    }

    pub fn material(&self, scene: &mut Scene) -> Result<Option<Handle<Material>>> {
        let idx = scene.mrenderer_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| scene.mrenderer_data.c.data[i].material)
    }

    pub fn layers(&self, scene: &Scene) -> Result<i32> {
        let idx = scene.mrenderer_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &scene.mrenderer_data.c.data[i];
                data.layers
            })
    }

    pub fn is_valid(&self, scene: &Scene) -> bool {
        scene.mrenderer_data.is_handle_valid(*self)
    }
}

impl Component for MeshRenderer {
    fn init(scene: &mut Scene, object: Handle<Object>) -> Result<Handle<MeshRenderer>> {
        scene.create_mrenderer(object)
    }

    fn marked(component: Handle<MeshRenderer>, scene: &Scene) -> Result<bool> {
        let idx = scene.mrenderer_data.data_idx_checked(component);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                scene.mrenderer_data.c.data[i].marked
            })
    }

    fn destroy(component: Handle<MeshRenderer>, scene: &mut Scene) {
        scene.destroy_mrenderer(component);
    }
}

pub trait BehaviourMessages: 'static {
    fn start(&self, scene: &mut Scene);

    fn update(&self, scene: &mut Scene);

    fn destroy(&self, scene: &mut Scene);
}

trait AnyBehaviour: Any + BehaviourMessages {
    fn as_any(&self) -> &Any;
}

impl<T: BehaviourMessages + 'static> AnyBehaviour for T {
    fn as_any(&self) -> &Any {
        self
    }
}

struct BehaviourData {
    /// Reference to the behaviour implementation.
    behaviour: Rc<AnyBehaviour>,
    /// True when the object has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    /// True if the object has been newly created this current frame.
    is_new: bool
}

struct BehaviourContainer {
    data: Vec<BehaviourData>,
}

impl Container for BehaviourContainer {
    type Item = BehaviourData;
    type HandleType = Behaviour;

    fn push(&mut self, value: Self::Item) {
        self.data.push(value);
    }

    fn swap_remove(&mut self, idx: usize) {
        self.data.swap_remove(idx);
    }
}

impl BehaviourContainer {
    fn new() -> BehaviourContainer {
        BehaviourContainer {
            data: Vec::new(),
        }
    }
}

/// A handle to a behaviour for a scene.
#[derive(Copy, Clone)]
pub struct Behaviour;

impl Handle<Behaviour> {
    pub fn behaviour<'a, T: Any>(&self, scene: &Scene) -> Result<Rc<T>> {
        let idx = scene.behaviour_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .and_then(|i| {
                let behaviour = &scene.behaviour_data.c.data[i].behaviour;
                if behaviour.as_any().is::<T>() {
                    unsafe {
                        let raw: *mut AnyBehaviour = &**behaviour as *const _ as *mut _;
                        std::mem::forget(behaviour);
                        Ok(Rc::from_raw(raw as *mut T))
                    }
                } else {
                    // TODO: dedicated error variant
                    Err(Error::Other)
                }
            })
    }

    pub fn is_valid(&self, scene: &Scene) -> bool {
        scene.behaviour_data.is_handle_valid(*self)
    }
}

struct ObjectData {
    /// The components attached to this object.
    components: HashMap<TypeId, GenericHandle>,
    /// True when the object has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    /// True if this object has had children marked for destruction.
    destroyed_children: bool,
}

struct TransformData {
    /// Local rotation.
    rot: Quaternion<f32>,
    /// Local position.
    pos: Vector3<f32>,
    /// List of handles of the children of this object.
    children: Vec<Handle<Object>>,
    /// Index of this object's parent.
    parent_idx: Option<usize>,
    /// Cached local-to-world matrix for this transform.
    ltw_matrix: Cell<[[f32; 4]; 4]>,
    /// False if our local rotation / position has been changed, or we've been
    /// reparented this frame. False will cause this transform and our child
    /// transforms to have their local-to-world matrix recomputed.
    ltw_valid: Cell<bool>,
}

struct ObjectContainer {
    obj_data: Vec<ObjectData>,
    trans_data: Vec<TransformData>,
}

impl Container for ObjectContainer {
    type Item = (ObjectData, TransformData);
    type HandleType = Object;

    fn push(&mut self, value: Self::Item) {
        self.obj_data.push(value.0);
        self.trans_data.push(value.1);
    }

    fn swap_remove(&mut self, idx: usize) {
        self.obj_data.swap_remove(idx);
        self.trans_data.swap_remove(idx);
    }
}

impl ObjectContainer {
    fn new() -> ObjectContainer {
        ObjectContainer {
            obj_data: Vec::new(),
            trans_data: Vec::new(),
        }
    }
}

/// A handle to a object for a scene.
#[derive(Copy, Clone)]
pub struct Object;

impl Handle<Object> {
    pub fn children<'a>(&self, scene: &'a Scene) -> Result<&'a [Handle<Object>]> {
        let idx = scene.object_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                &scene.object_data.c.trans_data[i].children[..]
            })
    }

    pub fn set_local_pos(&self, scene: &mut Scene, pos: Vector3<f32>) -> Result<()> {
        let idx = scene.object_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &mut scene.object_data.c.trans_data[i];
                data.pos = pos;
                data.ltw_valid.set(false);
            })
    }

    pub fn local_pos(&self, scene: &Scene) -> Result<Vector3<f32>> {
        let idx = scene.object_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| scene.object_data.c.trans_data[i].pos)
    }

    pub fn set_local_rot(&self, scene: &mut Scene, rot: Quaternion<f32>) -> Result<()> {
        let idx = scene.object_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &mut scene.object_data.c.trans_data[i];
                data.rot = rot;
                data.ltw_valid.set(false);
            })
    }

    pub fn local_rot(&self, scene: &Scene) -> Result<Quaternion<f32>> {
        let idx = scene.object_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| scene.object_data.c.trans_data[i].rot)
    }

    pub fn set_world_pos(&self, scene: &mut Scene, pos: Vector3<f32>) -> Result<()> {
        let idx = scene.object_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let local_pos = {
                    let data = &scene.object_data.c.trans_data[i];
                    let parent_data = data.parent_idx
                        .map(|idx| &scene.object_data.c.trans_data[idx]);
                    scene.world_to_local_pos(parent_data, pos)
                };
                let data = &mut scene.object_data.c.trans_data[i];
                data.pos = local_pos;
            })
    }

    pub fn world_pos(&self, scene: &Scene) -> Result<Vector3<f32>> {
        let idx = scene.object_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &scene.object_data.c.trans_data[i];
                let parent_data = data.parent_idx
                    .map(|idx| &scene.object_data.c.trans_data[idx]);
                scene.local_to_world_pos(parent_data, data.pos)
            })
    }

    pub fn set_world_rot(&self, scene: &mut Scene, rot: Quaternion<f32>) -> Result<()> {
        let idx = scene.object_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let local_rot = {
                    let data = &scene.object_data.c.trans_data[i];
                    let parent_data = data.parent_idx
                        .map(|idx| &scene.object_data.c.trans_data[idx]);
                    scene.world_to_local_rot(parent_data, rot)
                };
                let data = &mut scene.object_data.c.trans_data[i];
                data.rot = local_rot;
                data.ltw_valid.set(false);
            })
    }

    pub fn world_rot(&self, scene: &Scene) -> Result<Quaternion<f32>> {
        let idx = scene.object_data.data_idx_checked(*self);
        idx.or(Err(Error::ObjectDestroyed))
            .map(|i| {
                let data = &scene.object_data.c.trans_data[i];
                let parent_data = data.parent_idx
                    .map(|idx| &scene.object_data.c.trans_data[idx]);
                scene.local_to_world_rot(parent_data, data.rot)
            })
    }

    pub fn is_valid(&self, scene: &Scene) -> bool {
        scene.object_data.is_handle_valid(*self)
    }
}

const FRAME_TIME_MAX_SAMPLES: usize = 60;
const INITIAL_VERTEX_BUF_CAPACITY: usize = 4 * 1024 * 1024;
const INITIAL_INDEX_BUF_CAPACITY: usize = 4 * 1024 * 1024;
const MESH_MIN_VERT_CAPACITY: usize = 1;
const MESH_MIN_INDICES_CAPACITY: usize = 1;

pub struct Scene {
    /// The OpenGL used for rendering, None if in headless mode.
    ctx: Option<GLContext>,
    /// Whether we are running on Intel graphics, don't try anything remotely
    /// fancy if so.
    buggy_intel: bool,
    window: Option<Window>,
    event_pump: Option<EventPump>,
    camera_data: ObjectManager<CameraContainer>,
    mesh_data: Vec<MeshData>,
    cubemap_data: ObjectManager<CubemapContainer>,
    material_data: ObjectManager<MaterialContainer>,
    mrenderer_data: ObjectManager<MeshRendererContainer>,
    shader_data: ObjectManager<ShaderContainer>,
    behaviour_data: ObjectManager<BehaviourContainer>,
    object_data: ObjectManager<ObjectContainer>,
    destroyed_cameras: Vec<Handle<Camera>>,
    destroyed_meshes: Vec<usize>,
    destroyed_cubemaps: Vec<Handle<Cubemap>>,
    destroyed_materials: Vec<Handle<Material>>,
    destroyed_mrenderers: Vec<Handle<MeshRenderer>>,
    destroyed_shaders: Vec<Handle<Shader>>,
    destroyed_behaviours: Vec<Handle<Behaviour>>,
    destroyed_objects: Vec<Handle<Object>>,
    update_parents: Vec<usize>,

    vertex_bufs: Vec<VertexBuffer>,
    index_bufs: Vec<IndexBuffer>,

    /// Temporary vector used in `local_to_world_pos_rot()`
    tmp_vec: UnsafeCell<Vec<(Vector3<f32>, Quaternion<f32>)>>,

    cur_fence: usize,
    fences: [Option<GLsync>; 2],
    /// Temporary vectors used when freeing unused vertex buffer ranges.
    vertex_free_lists: [Vec<(usize, AllocRange)>; 2],
    /// Temporary vectors used when freeing unused index buffer ranges.
    index_free_lists: [Vec<(usize, AllocRange)>; 2],

    /// Times spent on rendering a frame, first value is in behaviour updates,
    /// second value is in drawing, third value is in object destruction
    frame_times: VecDeque<(f32, f32, f32)>,

    /// A vector of indices into `camera_data` describing the order in which to
    /// render each camera.
    camera_render_order: Vec<usize>,
}

struct VertexBuffer {
    alloc: SimpleAllocator,
    update_tasks: Vec<usize>,
    vao_id: GLuint,
    buf_id: GLuint,
}

struct IndexBuffer {
    alloc: SimpleAllocator,
    update_tasks: Vec<usize>,
    buf_id: GLuint,
}

impl VertexBuffer {
    fn new(vert_capacity: usize) -> VertexBuffer {
        let alloc = SimpleAllocator::new(vert_capacity);
        let buf_size = (vert_capacity * 9 * 4) as GLsizeiptr;
        let mut vao_id = 0;
        let mut buf_id = 0;

        unsafe {
            gl::GenVertexArrays(1, &mut vao_id);
            gl::GenBuffers(1, &mut buf_id);

            gl::BindVertexArray(vao_id);

            gl::BindBuffer(gl::ARRAY_BUFFER, buf_id);
            gl::BufferData(gl::ARRAY_BUFFER, buf_size, ptr::null(), gl::STATIC_DRAW);
            gl::VertexAttribPointer(0, 3, gl::FLOAT, gl::FALSE, 36, 0 as *const _);
            gl::EnableVertexAttribArray(0);

            gl::VertexAttribPointer(1, 3, gl::FLOAT, gl::FALSE, 36, 12 as *const _);
            gl::EnableVertexAttribArray(1);

            gl::VertexAttribPointer(2, 3, gl::FLOAT, gl::FALSE, 36, 24 as *const _);
            gl::EnableVertexAttribArray(2);

            gl::BindVertexArray(0);
        }

        VertexBuffer {
            alloc,
            update_tasks: Vec::new(),
            vao_id,
            buf_id,
        }
    }
}

impl IndexBuffer {
    fn new(indices_capacity: usize) -> IndexBuffer {
        let alloc = SimpleAllocator::new(indices_capacity);
        let buf_size = (indices_capacity * 2) as GLsizeiptr;
        let mut buf_id = 0;

        unsafe {
            gl::GenBuffers(1, &mut buf_id);

            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, buf_id);
            gl::BufferData(gl::ELEMENT_ARRAY_BUFFER, buf_size, ptr::null(), gl::STATIC_DRAW);
        }

        IndexBuffer {
            alloc,
            update_tasks: Vec::new(),
            buf_id,
        }
    }
}

impl Scene {
    pub fn new(sdl: Sdl, window_name: &str) -> Scene {
        let video_sys = sdl.video()
            .expect("Failed to initialize the Video subsystem");
        let event_pump = sdl.event_pump()
            .expect("Failed to obtain the SDL event pump");
        let window = video_sys.window(window_name, 800, 600)
            .position_centered()
            .resizable()
            .opengl()
            .build()
            .expect("Failed to create a window");

        video_sys.gl_attr().set_context_profile(GLProfile::Core);

        let ctx = window.gl_create_context()
            .expect("Failed to create GL context");

        gl::load_with(|name| video_sys.gl_get_proc_address(name) as *const _);

        let buggy_intel = unsafe {
            let vendor_cstr = CStr::from_ptr::<'static>(gl::GetString(gl::VENDOR) as *const i8);
            let vendor_str = vendor_cstr.to_str().expect("Failed to convert vendor string to UTF-8.");
            println!("[INFO] OpenGL vendor: {}", vendor_str);

            if vendor_str.contains("Intel") {
                true
            } else {
                false
            }
        };

        // V-Sync
        video_sys.gl_set_swap_interval(1);

        let mut scene = Scene {
            ctx: Some(ctx),
            buggy_intel,
            window: Some(window),
            event_pump: Some(event_pump),
            camera_data: ObjectManager::new(CameraContainer::new()),
            mesh_data: Vec::new(),
            cubemap_data: ObjectManager::new(CubemapContainer::new()),
            material_data: ObjectManager::new(MaterialContainer::new()),
            mrenderer_data: ObjectManager::new(MeshRendererContainer::new()),
            shader_data: ObjectManager::new(ShaderContainer::new()),
            behaviour_data: ObjectManager::new(BehaviourContainer::new()),
            object_data: ObjectManager::new(ObjectContainer::new()),
            destroyed_cameras: Vec::new(),
            destroyed_meshes: Vec::new(),
            destroyed_cubemaps: Vec::new(),
            destroyed_materials: Vec::new(),
            destroyed_mrenderers: Vec::new(),
            destroyed_shaders: Vec::new(),
            destroyed_behaviours: Vec::new(),
            destroyed_objects: Vec::new(),
            update_parents: Vec::new(),

            vertex_bufs: Vec::new(),
            index_bufs: Vec::new(),

            cur_fence: 0,
            fences: [None, None],
            vertex_free_lists: [Vec::new(), Vec::new()],
            index_free_lists: [Vec::new(), Vec::new()],

            tmp_vec: UnsafeCell::new(Vec::new()),
            frame_times: VecDeque::with_capacity(FRAME_TIME_MAX_SAMPLES),

            camera_render_order: Vec::new(),
        };

        Scene::create_new_vertex_buffer(&mut scene.vertex_bufs);
        Scene::create_new_index_buffer(&mut scene.index_bufs);

        scene
    }

    /// Creates a new scene in "headless" mode (no graphical capabilities). Many
    /// operations will not work currently.
    pub fn new_headless() -> Scene {
        Scene {
            ctx: None,
            buggy_intel: false,
            window: None,
            event_pump: None,
            camera_data: ObjectManager::new(CameraContainer::new()),
            mesh_data: Vec::new(),
            cubemap_data: ObjectManager::new(CubemapContainer::new()),
            material_data: ObjectManager::new(MaterialContainer::new()),
            mrenderer_data: ObjectManager::new(MeshRendererContainer::new()),
            shader_data: ObjectManager::new(ShaderContainer::new()),
            behaviour_data: ObjectManager::new(BehaviourContainer::new()),
            object_data: ObjectManager::new(ObjectContainer::new()),
            destroyed_cameras: Vec::new(),
            destroyed_meshes: Vec::new(),
            destroyed_cubemaps: Vec::new(),
            destroyed_materials: Vec::new(),
            destroyed_mrenderers: Vec::new(),
            destroyed_shaders: Vec::new(),
            destroyed_behaviours: Vec::new(),
            destroyed_objects: Vec::new(),
            update_parents: Vec::new(),

            index_bufs: Vec::new(),
            vertex_bufs: Vec::new(),

            cur_fence: 0,
            fences: [None, None],
            vertex_free_lists: [Vec::new(), Vec::new()],
            index_free_lists: [Vec::new(), Vec::new()],

            tmp_vec: UnsafeCell::new(Vec::new()),
            frame_times: VecDeque::with_capacity(FRAME_TIME_MAX_SAMPLES),

            camera_render_order: Vec::new(),
        }
    }

    fn create_new_vertex_buffer(vertex_bufs: &mut Vec<VertexBuffer>) {
        let vertex_buf = VertexBuffer::new(INITIAL_VERTEX_BUF_CAPACITY);
        vertex_bufs.push(vertex_buf);
    }

    fn create_new_index_buffer(index_bufs: &mut Vec<IndexBuffer>) {
        let index_buf = IndexBuffer::new(INITIAL_INDEX_BUF_CAPACITY);
        index_bufs.push(index_buf);
    }

    fn create_camera(&mut self, object: Handle<Object>) -> Result<Handle<Camera>> {
        if let Err(_) = self.object_data.data_idx_checked(object) {
            return Err(Error::ObjectDestroyed);
        }

        let handle = self.camera_data.add(CameraData {
            parent: object,
            enabled: true,
            marked: false,
            order: 0,
            layers: 1,
            fovy: Deg(90.0),
            aspect: 1.0,
            near_clip: 1.0,
            far_clip: 1000.0,
            pos: Vector3::zero(),
            pv_matrix: Matrix4::one(),
            draw_cmds: Vec::new(),
        });

        Ok(handle)
    }

    pub fn create_mesh(&mut self, vert_capacity: usize, indices_capacity: usize) -> Rc<Mesh> {
        if self.ctx.is_none() {
            // TODO: In the future implement some kind of dummy mesh?
            panic!("Tried to create mesh in headless mode.");
        }

        let vert_capacity = std::cmp::max(vert_capacity, MESH_MIN_VERT_CAPACITY);
        let indices_capacity = std::cmp::max(indices_capacity, MESH_MIN_INDICES_CAPACITY);

        let rv = Rc::new(Mesh { idx: Cell::new(None) });
        let data = MeshData {
            object: rv.clone(),
            marked: false,
            vpos_vec: Vec::new(),
            vnorm_vec: Vec::new(),
            vcolour_vec: Vec::new(),
            indices_vec: Vec::new(),
            vertex_buf_dirty: true,
            vertex_buf_alloc: None,
            vertex_buf_capacity: vert_capacity,
            index_buf_dirty: true,
            index_buf_alloc: None,
            index_buf_capacity: indices_capacity,
            aabb_dirty: false,
            aabb_x: [0.0, 0.0],
            aabb_y: [0.0, 0.0],
            aabb_z: [0.0, 0.0],
        };
        self.mesh_data.push(data);
        rv.idx.set(Some(self.mesh_data.len() - 1));
        rv
    }

    pub fn create_cubemap(&mut self, width: usize, height: usize, faces: [&[u8]; 6]) -> Handle<Cubemap> {
        for f in &faces {
            let size = width.checked_mul(height).and_then(|x| x.checked_mul(3)).unwrap();
            assert_eq!(size, f.len());
            assert!(f.len() <= std::isize::MAX as usize);
        }

        let mut texture_id = 0;
        unsafe {
            gl::GenTextures(1, &mut texture_id);
            gl::BindTexture(gl::TEXTURE_CUBE_MAP, texture_id);

            for (i, f) in faces.iter().enumerate() {
                gl::TexImage2D(
                    gl::TEXTURE_CUBE_MAP_POSITIVE_X + i as u32,
                    0, gl::RGB as GLint, width as GLsizei, height as GLsizei, 0, gl::RGB, gl::UNSIGNED_BYTE, f.as_ptr() as *const _);
            }

            gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_MAG_FILTER, gl::LINEAR as GLint);
            gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_MIN_FILTER, gl::LINEAR as GLint);
            gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as GLint);
            gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as GLint);
            gl::TexParameteri(gl::TEXTURE_CUBE_MAP, gl::TEXTURE_WRAP_R, gl::CLAMP_TO_EDGE as GLint);
        }

        self.cubemap_data.add(CubemapData {
            marked: false,
            texture_id,
        })
    }

    pub fn create_material(&mut self, shader: Handle<Shader>) -> Result<Handle<Material>> {
        let (map, uniform_values) = {
            let shader_data = try!(
                self.shader_data.data_idx_checked(shader)
                    .or(Err(Error::ObjectDestroyed))
                    .map(|i| &self.shader_data.c.data[i]));

            let mut map = HashMap::new();
            let mut uniform_values = Vec::new();

            let mut i = 0;
            for (name, location) in shader_data.program.uniforms() {
                // Names starting with "_" are reserved for our own use
                if !name.starts_with("_") {
                    map.insert(name, i);
                    uniform_values.push((location, None));
                    i += 1;
                }
            }

            (map, uniform_values)
        };

        let handle = self.material_data.add(MaterialData {
            marked: false,
            uniforms: map,
            uniform_values,
            shader: shader,
            colour: (1.0, 1.0, 1.0)
        });
        Ok(handle)
    }

    fn create_mrenderer(&mut self, object: Handle<Object>) -> Result<Handle<MeshRenderer>> {
        if let Err(_) = self.object_data.data_idx_checked(object) {
            return Err(Error::ObjectDestroyed);
        }

        let handle = self.mrenderer_data.add(MeshRendererData {
            parent: object,
            marked: false,
            enabled: true,
            // By default, render on the first layer only
            layers: 1,
            mesh: None,
            material: None,
        });

        Ok(handle)
    }

    pub fn create_shader(&mut self, vs_src: &str, fs_src: &str, _gs_src: Option<&str>) -> Handle<Shader> {
        if self.ctx.is_none() {
            // TODO: In the future implement some kind of dummy shader?
            panic!("Tried to create shader in headless mode.");
        }

        let program = Program::new(vs_src, fs_src).unwrap();
        let obj_matrix_uniform = unsafe {
            gl::GetUniformLocation(program.program, "_obj_matrix\0".as_ptr() as *const GLchar)
        };
        let cam_pos_uniform = unsafe {
            gl::GetUniformLocation(program.program, "_cam_pos\0".as_ptr() as *const GLchar)
        };
        let cam_matrix_uniform = unsafe {
            gl::GetUniformLocation(program.program, "_cam_matrix\0".as_ptr() as *const GLchar)
        };
        let colour_uniform = unsafe {
            gl::GetUniformLocation(program.program, "_colour\0".as_ptr() as *const GLchar)
        };

        self.shader_data.add(ShaderData {
            marked: false,
            program: program,
            obj_matrix_uniform: obj_matrix_uniform,
            cam_pos_uniform: cam_pos_uniform,
            cam_matrix_uniform: cam_matrix_uniform,
            colour_uniform: colour_uniform,
        })
    }

    pub fn create_behaviour<T: BehaviourMessages>(&mut self, t: T) -> Result<Handle<Behaviour>> {
        let rv = Rc::new(t);
        let handle = self.behaviour_data.add(BehaviourData {
            behaviour: rv.clone(),
            marked: false,
            is_new: true,
        });

        Ok(handle)
    }

    pub fn add_component<T: Component>(&mut self, object: Handle<Object>) -> Result<Handle<T>> {
        let obj_idx = match self.object_data.data_idx_checked(object) {
            Ok(obj_idx) => obj_idx,
            Err(_) => return Err(Error::ObjectDestroyed),
        };

        let id = TypeId::of::<T>();

        {
            let obj_data = &self.object_data.c.obj_data[obj_idx];

            if obj_data.marked {
                // TODO: dedicated error variant
                return Err(Error::Other)
            }
            // Only overwrite if it doesn't exist or is marked
            let overwrite = match obj_data.components.get(&id) {
                Some(&generic_handle) => {
                    let handle = Handle::from_generic_handle(generic_handle)
                        .expect("Conversion from generic handle failed.");
                    T::marked(handle, self) != Ok(false)
                }
                None => {
                    true
                }
            };
            if !overwrite {
                // TODO: dedicated error variant
                return Err(Error::Other);
            }

            if obj_data.marked {
                // TODO: dedicated error variant
                return Err(Error::Other);
            }
        }

        let comp = try!(T::init(self, object));
        let generic_comp = Handle::into_generic_handle(comp);

        let obj_data = &mut self.object_data.c.obj_data[obj_idx];
        obj_data.components.insert(id, generic_comp);

        Ok(comp)
    }

    pub fn get_component<T: Component>(&self, object: Handle<Object>) -> Result<Handle<T>> {
        let obj_idx = match self.object_data.data_idx_checked(object) {
            Ok(obj_idx) => obj_idx,
            Err(_) => return Err(Error::ObjectDestroyed),
        };

        let id = TypeId::of::<T>();

        let generic_comp = match self.object_data.c.obj_data[obj_idx].components.get(&id) {
            Some(&generic_comp) => generic_comp,
            None => {
                // TODO: dedicated error variant
                // Component doesn't exist
                return Err(Error::Other);
            }
        };

        let comp = Handle::from_generic_handle(generic_comp)
            .expect("Conversion from generic handle failed.");
        if T::marked(comp, self).is_err() {
            // TODO: dedicated error variant
            // Component doesn't exist
            Err(Error::Other)
        } else {
            Ok(comp)
        }
    }

    pub fn create_object(&mut self) -> Handle<Object> {
        let obj_data = ObjectData {
            components: HashMap::new(),
            marked: false,
            destroyed_children: false,
        };
        let trans_data = TransformData {
            rot: Quaternion::from(Euler {
                x: Deg(0.0),
                y: Deg(0.0),
                z: Deg(0.0)
            }),
            pos: Vector3::new(0.0, 0.0, 0.0),
            children: Vec::new(),
            parent_idx: None,
            ltw_matrix: Cell::new([[0.0; 4]; 4]),
            ltw_valid: Cell::new(false),
        };

        self.object_data.add((obj_data, trans_data))
    }

    pub fn destroy_camera(&mut self, camera: Handle<Camera>) {
        let camera_idx = match self.camera_data.data_idx_checked(camera) {
            Ok(camera_idx) => camera_idx,
            Err(_) => {
                println!("[WARNING] destroy_camera called on a camera without a valid handle!");
                return;
            }
        };

        let camera_data = &mut self.camera_data.c.data[camera_idx];

        if !camera_data.marked {
            self.destroyed_cameras.push(camera);
            camera_data.marked = true;
        }
    }

    pub fn destroy_mesh(&mut self, mesh: &Mesh) {
        let mesh_idx = match mesh.idx.get() {
            Some(mesh_idx) => mesh_idx,
            None => {
                println!("[WARNING] destroy_mesh called on a mesh without a valid handle!");
                return
            }
        };
        let mesh_data = unsafe {
            self.mesh_data.get_unchecked_mut(mesh_idx)
        };

        if !mesh_data.marked {
            self.destroyed_meshes.push(mesh_idx);
            mesh_data.marked = true;
        }
    }

    pub fn destroy_cubemap(&mut self, cubemap: Handle<Cubemap>) {
        let cubemap_idx = match self.cubemap_data.data_idx_checked(cubemap) {
            Ok(cubemap_idx) => cubemap_idx,
            Err(_) => {
                println!("[WARNING] destroy_cubemap called on a cubemap without a valid handle!");
                return;
            }
        };

        let cubemap_data = &mut self.cubemap_data.c.data[cubemap_idx];

        if !cubemap_data.marked {
            self.destroyed_cubemaps.push(cubemap);
            cubemap_data.marked = true;
        }
    }

    pub fn destroy_material(&mut self, material: Handle<Material>) {
        let material_idx = match self.material_data.data_idx_checked(material) {
            Ok(material_idx) => material_idx,
            Err(_) => {
                println!("[WARNING] destroy_material called on a material without a valid handle!");
                return;
            }
        };

        let material_data = &mut self.material_data.c.data[material_idx];

        if !material_data.marked {
            self.destroyed_materials.push(material);
            material_data.marked = true;
        }
    }

    pub fn destroy_mrenderer(&mut self, mrenderer: Handle<MeshRenderer>) {
        let mrenderer_idx = match self.mrenderer_data.data_idx_checked(mrenderer) {
            Ok(mrenderer_idx) => mrenderer_idx,
            Err(_) => {
                println!("[WARNING] destroy_mrenderer called on a mesh renderer without a valid handle!");
                return;
            }
        };
        let mrenderer_data = &mut self.mrenderer_data.c.data[mrenderer_idx];

        if !mrenderer_data.marked {
            self.destroyed_mrenderers.push(mrenderer);
            mrenderer_data.marked = true;
        }
    }

    pub fn destroy_shader(&mut self, shader: Handle<Shader>) {
        let shader_idx = match self.shader_data.data_idx_checked(shader) {
            Ok(shader_idx) => shader_idx,
            Err(_) => {
                println!("[WARNING] destroy_shader called on a mesh renderer without a valid handle!");
                return;
            }
        };
        let shader_data = &mut self.shader_data.c.data[shader_idx];

        if !shader_data.marked {
            self.destroyed_shaders.push(shader);
            shader_data.marked = true;
        }
    }

    pub fn destroy_behaviour(&mut self, behaviour: Handle<Behaviour>) {
        let bhav_idx = match self.behaviour_data.data_idx_checked(behaviour) {
            Ok(bhav_idx) => bhav_idx,
            Err(_) => {
                println!("[WARNING] destroy_behaviour called on a behaviour without a valid handle!");
                return;
            }
        };

        let bhav_data = &mut self.behaviour_data.c.data[bhav_idx];

        if !bhav_data.marked {
            self.destroyed_behaviours.push(behaviour);
            bhav_data.marked = true;
        }
    }

    pub fn destroy_object(&mut self, object: Handle<Object>) {
        let obj_idx = match self.object_data.data_idx_checked(object) {
            Ok(obj_idx) => obj_idx,
            Err(_) => {
                println!("[WARNING] destroy_object called on an object without a valid handle!");
                return;
            }
        };

        let parent_idx = self.object_data.c.trans_data[obj_idx].parent_idx;
        if let Some(parent_idx) = parent_idx {
            if !self.object_data.c.obj_data[parent_idx].destroyed_children {
                self.object_data.c.obj_data[parent_idx].destroyed_children = true;
                self.update_parents.push(parent_idx);
            }
        }

        self.destroy_object_internal(obj_idx);
    }

    fn destroy_object_internal(&mut self, idx: usize) {
        let (was_marked, mut components) = {
            // FIXME: This is one example of many of workarounds to placate the
            // borrow checker. If I understand correctly, non-lexically based
            // lifetimes based on liveness should help in most cases. Update the
            // code when NLL is implemented in Rust.
            let obj_data = &mut self.object_data.c.obj_data[idx];
            // Swap map to placate the borrow checker
            let mut components = HashMap::new();
            std::mem::swap(&mut components, &mut obj_data.components);
            let was_marked = obj_data.marked;
            obj_data.marked = true;
            (was_marked, components)
        };

        for (&k, &v) in &components {
            // XXX: Every component type here needs to be handled otherwise a
            // panic may occur.
            match k {
                t if t == TypeId::of::<Handle<Camera>>() => {
                    let v = Handle::<Camera>::from_generic_handle(v).unwrap();
                    Component::destroy(v, self);
                }
                t if t == TypeId::of::<Handle<MeshRenderer>>() => {
                    let v = Handle::<Camera>::from_generic_handle(v).unwrap();
                    Component::destroy(v, self);
                }
                t => panic!("Unhandled type: {:?}", t),
            }
        }

        std::mem::swap(&mut components, &mut self.object_data.c.obj_data[idx].components);

        // Swap vectors to placate the borrow checker
        let mut tmp_children = Vec::new();
        std::mem::swap(&mut tmp_children, &mut self.object_data.c.trans_data[idx].children);

        if !was_marked {
            self.destroyed_objects.push(self.object_data.handle(idx));
            for &child in &tmp_children {
                let child_idx = self.object_data.data_idx_checked(child)
                    .expect("Destroyed object found in hierarchy");
                Scene::destroy_object_internal(self, child_idx);
            }
        }

        std::mem::swap(&mut tmp_children, &mut self.object_data.c.trans_data[idx].children);
    }

    fn debug_check(&self) {
        // For all objects, check the following:
        //   * The index is valid (i.e. `is_some()`)
        //   * The index corresponds to the correct data entry
        /*for i in 0..self.camera_data.len() {
            let data = unsafe { self.camera_data.get_unchecked(i) };
            let idx = data.camera.idx.get();
            assert!(idx.is_some(), "Invalid object handle found!");
            assert_eq!(idx.unwrap(), i);
        }*/
        // TODO: move to object manager
        /*for i in 0..self.object_data.len() {
            let data = unsafe { self.object_data.get_unchecked(i) };
            let idx = data.object.idx.get();
            assert!(idx.is_some(), "Invalid object handle found!");
            assert_eq!(idx.unwrap(), i);
        }*/
    }

    pub fn do_frame(&mut self) -> bool {
        if cfg!(debug_assertions) {
            self.debug_check();
        }

        let start_time = Instant::now();

        let mut idx = 0;
        while idx < self.behaviour_data.c.data.len() {
            let idx = post_add(&mut idx, 1);
            unsafe {
                let (is_new, obj) = {
                    let data = &self.behaviour_data.c.data[idx];
                    // Don't run `update()` on destroyed behaviours
                    if data.marked {
                        println!("Skipping behaviour {} because it's marked.", idx);
                        continue
                    }
                    (data.is_new, &*data.behaviour as *const AnyBehaviour)
                };

                if is_new {
                    (*obj).start(self);
                    let marked = {
                        let data = &mut self.behaviour_data.c.data[idx];
                        data.is_new = false;
                        data.marked
                    };
                    // Check that the start function didn't immediately destroy the behaviour
                    if !marked {
                        (*obj).update(self);
                    }
                } else {
                    (*obj).update(self);
                }
            }
        }

        let mut i = 0;
        while i < self.destroyed_behaviours.len() {
            let i = post_add(&mut i, 1);
            let handle = self.destroyed_behaviours[i];
            let idx = self.behaviour_data.data_idx_checked(handle)
                .expect("Behaviour already destroyed");
            unsafe {
                let obj = {
                    let data = &self.behaviour_data.c.data[idx];
                    &*data.behaviour as *const AnyBehaviour
                };
                (*obj).destroy(self);
            }
        }

        let destroy_start_time = Instant::now();

        unsafe {
            for &handle in &self.destroyed_behaviours {
                self.behaviour_data.remove(handle).expect("Double free");
            }
            for &handle in &self.destroyed_cameras {
                self.camera_data.remove(handle).expect("Double free");
            }
            for &handle in &self.destroyed_mrenderers {
                self.mrenderer_data.remove(handle).expect("Double free");
            }
            self.destroyed_behaviours.clear();
            self.destroyed_cameras.clear();
            self.destroyed_mrenderers.clear();

            for &obj_idx in &self.update_parents {
                {
                    let obj_data = &self.object_data.c.obj_data;

                    let mut tmp_children = Vec::new();
                    std::mem::swap(&mut tmp_children, &mut self.object_data.c.trans_data[obj_idx].children);

                    if obj_data[obj_idx].destroyed_children && !obj_data[obj_idx].marked {
                        // Remove children marked for destruction, leaving all
                        // marked objects separate from non-marked objects.
                        tmp_children.retain(|&child| {
                            let child_idx = self.object_data.data_idx_checked(child)
                                .expect("Destroyed object found in hierarchy");
                            let child_obj_data = &obj_data[child_idx];
                            !child_obj_data.marked
                        });
                    }

                    std::mem::swap(&mut tmp_children, &mut self.object_data.c.trans_data[obj_idx].children);
                }
                self.object_data.c.obj_data[obj_idx].destroyed_children = true;
            }
            self.update_parents.clear();

            self.cleanup_destroyed_objects();
            // FIXME: resource leak
            Scene::cleanup_destroyed(
                &mut self.mesh_data, &mut self.destroyed_meshes,
                |x| x.marked,
                |x, idx| x.object.idx.set(idx));
            for &handle in &self.destroyed_cubemaps {
                let idx = self.cubemap_data.data_idx_checked(handle)
                    .expect("Double free");
                gl::DeleteTextures(1, &self.cubemap_data.c.data[idx].texture_id);
                self.cubemap_data.remove_idx(idx);
            }
            self.destroyed_cubemaps.clear();
            for &handle in &self.destroyed_materials {
                self.material_data.remove(handle).expect("Double free");
            }
            self.destroyed_materials.clear();
            // FIXME: resource leak
            for &handle in &self.destroyed_shaders {
                self.shader_data.remove(handle).expect("Double free");
            }
        }

        let draw_start_time = Instant::now();
        if self.window.is_some() {
            unsafe {
                if !self.draw() {
                    return false
                }
            }
        }
        let draw_end_time = Instant::now();

        //// Just block for until a new line is received
        //println!("Enter newline to continue...");
        //let mut _tmp = String::new();
        //std::io::stdin().read_line(&mut _tmp).unwrap();

        let update_time = destroy_start_time - start_time;
        let draw_time = draw_end_time - draw_start_time;
        let destroy_time = draw_start_time - destroy_start_time;

        let update_time_millis = (update_time.as_secs() as f32) * 1000.0 + update_time.subsec_nanos() as f32 / 1000000.0;
        let draw_time_millis = (draw_time.as_secs() as f32) * 1000.0 + draw_time.subsec_nanos() as f32 / 1000000.0;
        let destroy_time_millis = (destroy_time.as_secs() as f32) * 1000.0 + destroy_time.subsec_nanos() as f32 / 1000000.0;

        if self.frame_times.len() == FRAME_TIME_MAX_SAMPLES {
            self.frame_times.pop_back();
        }

        self.frame_times.push_front((update_time_millis, draw_time_millis, destroy_time_millis));

        true
    }

    pub fn print_frame_stats(&self) {
        let (mut min_update_time, mut max_update_time, mut sum_update_time) =
            (std::f32::MAX, std::f32::MIN, 0.0);
        let (mut min_draw_time, mut max_draw_time, mut sum_draw_time) =
            (std::f32::MAX, std::f32::MIN, 0.0);
        let (mut min_destroy_time, mut max_destroy_time, mut sum_destroy_time) =
            (std::f32::MAX, std::f32::MIN, 0.0);
        let (mut min_total_time, mut max_total_time, mut sum_total_time) =
            (std::f32::MAX, std::f32::MIN, 0.0);

        for &(update_time, draw_time, destroy_time) in &self.frame_times {
            let total_time = update_time + draw_time + destroy_time;

            min_update_time = f32::min(min_update_time, update_time);
            max_update_time = f32::max(max_update_time, update_time);
            sum_update_time += update_time;

            min_draw_time = f32::min(min_draw_time, draw_time);
            max_draw_time = f32::max(max_draw_time, draw_time);
            sum_draw_time += draw_time;

            min_destroy_time = f32::min(min_destroy_time, destroy_time);
            max_destroy_time = f32::max(max_destroy_time, destroy_time);
            sum_destroy_time += destroy_time;

            min_total_time = f32::min(min_total_time, total_time);
            max_total_time = f32::max(max_total_time, total_time);
            sum_total_time += total_time;
        }

        let avg_update_time = sum_update_time / self.frame_times.len() as f32;
        let avg_draw_time = sum_draw_time / self.frame_times.len() as f32;
        let avg_destroy_time = sum_destroy_time / self.frame_times.len() as f32;
        let avg_total_time = sum_total_time / self.frame_times.len() as f32;

        println!("Update:  min = {} ms, max = {} ms, avg = {} ms",
                 min_update_time, max_update_time, avg_update_time);
        println!("Draw:    min = {} ms, max = {} ms, avg = {} ms",
                 min_draw_time, max_draw_time, avg_draw_time);
        println!("Destroy: min = {} ms, max = {} ms, avg = {} ms",
                 min_destroy_time, max_destroy_time, avg_destroy_time);
        println!("Total:   min = {} ms, max = {} ms, avg = {} ms",
                 min_total_time, max_total_time, avg_total_time);
        println!("FPS:     min = {}, max = {}, avg = {}",
                 1000.0 / max_total_time, 1000.0 / min_total_time, 1000.0 / avg_total_time);
    }

    fn get_alloc_buffers(&mut self, idx: usize) -> (Option<(usize, usize)>, Option<(usize, usize)>) {
        let data = &self.mesh_data[idx];
        let vertex_buf_len = data.vertex_buf_len();
        let indices_buf_len = data.indices_vec.len();

        let vb_needs_alloc = data.vertex_buf_dirty;
        let ib_needs_alloc = data.index_buf_dirty;

        let vert_capacity = if vertex_buf_len > data.vertex_buf_capacity {
            vertex_buf_len * 2
        } else {
            data.vertex_buf_capacity
        };
        let indices_capacity = if indices_buf_len > data.index_buf_capacity {
            indices_buf_len * 2
        } else {
            data.index_buf_capacity
        };

        if ib_needs_alloc && vb_needs_alloc {
            // Try to allocate in same buffer first
            let (mut vb_idx, mut ib_idx) = (None, None);
            for i in 0..std::cmp::max(self.vertex_bufs.len(), self.index_bufs.len()) {
                let vb = &self.vertex_bufs.get(i);
                let ib = &self.index_bufs.get(i);

                let vb_ok = vb.map_or(false, |vb| vb.alloc.max_alloc() >= vert_capacity);
                let ib_ok = ib.map_or(false, |ib| ib.alloc.max_alloc() >= indices_capacity);

                if vb_ok && ib_ok {
                    vb_idx = Some(i);
                    ib_idx = Some(i);
                    break;
                } else if vb_ok && vb_idx.is_none() {
                    vb_idx = Some(i);
                } else if ib_ok && ib_idx.is_none() {
                    ib_idx = Some(i);
                }
            }

            let (vb_idx, ib_idx) = match (vb_idx, ib_idx) {
                (Some(vb_idx), Some(ib_idx)) => (vb_idx, ib_idx),
                (Some(vb_idx), None) => {
                    // No index buffer has space, need to allocate new index buffer
                    if self.index_bufs.len() < self.vertex_bufs.len() {
                        // Check for an opportunity for index and vertex
                        // buffers to use the same index.
                        let ib_idx = self.index_bufs.len();
                        let vb = &self.vertex_bufs[ib_idx];
                        let vb_idx = if vb.alloc.max_alloc() >= vert_capacity {
                            ib_idx
                        } else {
                            vb_idx
                        };
                        Scene::create_new_index_buffer(&mut self.index_bufs);
                        (vb_idx, ib_idx)
                    } else {
                        let ib_idx = self.index_bufs.len();
                        Scene::create_new_index_buffer(&mut self.index_bufs);
                        (vb_idx, ib_idx)
                    }
                },
                (None, Some(ib_idx)) => {
                    // No vertex buffer has space, need to allocate new vertex buffer
                    if self.vertex_bufs.len() < self.index_bufs.len() {
                        // Check for an opportunity for index and vertex
                        // buffers to use the same index.
                        let vb_idx = self.vertex_bufs.len();
                        let ib = &self.index_bufs[vb_idx];
                        let ib_idx = if ib.alloc.max_alloc() >= indices_capacity {
                            vb_idx
                        } else {
                            ib_idx
                        };
                        Scene::create_new_vertex_buffer(&mut self.vertex_bufs);
                        (vb_idx, ib_idx)
                    } else {
                        let vb_idx = self.vertex_bufs.len();
                        Scene::create_new_vertex_buffer(&mut self.vertex_bufs);
                        (vb_idx, ib_idx)
                    }
                },
                (None, None) => {
                    // No space, need to allocate new buffers
                    let (vb_idx, ib_idx) = (self.vertex_bufs.len(), self.index_bufs.len());
                    Scene::create_new_vertex_buffer(&mut self.vertex_bufs);
                    Scene::create_new_index_buffer(&mut self.index_bufs);
                    (vb_idx, ib_idx)
                },
            };

            (Some((vb_idx, vert_capacity)), Some((ib_idx, indices_capacity)))
        } else if ib_needs_alloc {
            let vb_idx = data.vertex_buf_alloc
                .as_ref()
                .map(|&(ref idx, _)| *idx)
                .expect("Expected vertex buffer to be already allocated");
            let mut ib_idx = None;
            for i in 0..self.index_bufs.len() {
                let ib = &self.index_bufs[i];

                let ib_ok = ib.alloc.max_alloc() >= indices_capacity;

                if ib_ok && i == vb_idx {
                    // Allocate in same buffer as vertex data
                    ib_idx = Some(i);
                    break;
                } else if ib_ok && ib_idx.is_none() {
                    ib_idx = Some(i);
                }
            }

            let ib_idx = match ib_idx {
                Some(ib_idx) => ib_idx,
                None => {
                    // No index buffer has space, need to allocate new index buffer
                    let ib_idx = self.index_bufs.len();
                    Scene::create_new_index_buffer(&mut self.index_bufs);
                    ib_idx
                }
            };

            (None, Some((ib_idx, indices_capacity)))
        } else if vb_needs_alloc {
            let ib_idx = data.index_buf_alloc
                .as_ref()
                .map(|&(ref idx, _)| *idx)
                .expect("Expected index buffer to be already allocated");
            let mut vb_idx = None;
            for i in 0..self.vertex_bufs.len() {
                let vb = &self.vertex_bufs[i];

                let vb_ok = vb.alloc.max_alloc() >= vert_capacity;

                if vb_ok && i == ib_idx {
                    // Allocate in same buffer as index data
                    vb_idx = Some(i);
                    break;
                } else if vb_ok && vb_idx.is_none() {
                    vb_idx = Some(i);
                }
            }

            let vb_idx = match vb_idx {
                Some(vb_idx) => vb_idx,
                None => {
                    // No index buffer has space, need to allocate new index buffer
                    let vb_idx = self.vertex_bufs.len();
                    Scene::create_new_vertex_buffer(&mut self.vertex_bufs);
                    vb_idx
                }
            };

            (Some((vb_idx, vert_capacity)), None)
        } else {
            (None, None)
        }
    }

    pub unsafe fn draw(&mut self) -> bool {
        gl::Enable(gl::CULL_FACE);
        gl::Enable(gl::DEPTH_TEST);
        gl::DepthFunc(gl::LESS);
        gl::ClearColor(0.8, 0.8, 0.8, 1.0);
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);

        {
            let (width, height) = self.window.as_ref().unwrap().size();
            let aspect_ratio = width as f32 / height as f32;
            for camera in &mut self.camera_data.c.data {
                camera.aspect = aspect_ratio;
            }
            gl::Viewport(0, 0, width as GLint, height as GLint);
        }

        if let Some(fence) = self.fences[self.cur_fence].take() {
            if !self.buggy_intel {
                let result = gl::ClientWaitSync(fence, gl::SYNC_FLUSH_COMMANDS_BIT,
                                                365 * 24 * 3600 * 1000 * 1000 * 1000);
                match result {
                    gl::ALREADY_SIGNALED | gl::CONDITION_SATISFIED => (),
                    _ => panic!("glClientWaitSync failed unexpectedly."),
                };
                gl::DeleteSync(fence);
            }

            for (vb_idx, range) in self.vertex_free_lists[self.cur_fence].drain(..) {
                self.vertex_bufs[vb_idx].alloc.free(range);
            }

            for (ib_idx, range) in self.index_free_lists[self.cur_fence].drain(..) {
                self.index_bufs[ib_idx].alloc.free(range);
            }
        }

        let mut vertex_free_list = Vec::new();
        let mut index_free_list = Vec::new();

        std::mem::swap(&mut vertex_free_list, &mut self.vertex_free_lists[self.cur_fence]);
        std::mem::swap(&mut index_free_list, &mut self.index_free_lists[self.cur_fence]);

        // Reallocate mesh buffers if necessary
        for i in 0..self.mesh_data.len() {
            let (vb_alloc, ib_alloc) = self.get_alloc_buffers(i);
            let data = &mut self.mesh_data[i];

            if let Some((vb_idx, vert_capacity)) = vb_alloc {
                // Free a previous allocation if there was one
                data.vertex_buf_alloc.take().map(|alloc| {
                    vertex_free_list.push(alloc);
                });

                let range = self.vertex_bufs[vb_idx].alloc.alloc(vert_capacity)
                    .expect("unexpected allocation failure");

                data.vertex_buf_alloc = Some((vb_idx, range));
            }

            if let Some((ib_idx, indices_capacity)) = ib_alloc {
                // Free a previous allocation if there was one
                data.index_buf_alloc.take().map(|alloc| {
                    index_free_list.push(alloc);
                });

                let range = self.index_bufs[ib_idx].alloc.alloc(indices_capacity)
                    .expect("unexpected allocation failure");

                data.index_buf_alloc = Some((ib_idx, range));
            }
        }

        std::mem::swap(&mut vertex_free_list, &mut self.vertex_free_lists[self.cur_fence]);
        std::mem::swap(&mut index_free_list, &mut self.index_free_lists[self.cur_fence]);

        for (i, data) in self.mesh_data.iter().enumerate() {
            if data.vertex_buf_dirty {
                let vertex_buf_idx = data.vertex_buf_alloc
                    .as_ref()
                    .map(|&(ref idx, _)| *idx)
                    .expect("Expected vertex buffer to be already allocated");

                self.vertex_bufs[vertex_buf_idx].update_tasks.push(i);
            }
            if data.index_buf_dirty {
                let index_buf_idx = data.index_buf_alloc
                    .as_ref()
                    .map(|&(ref idx, _)| *idx)
                    .expect("Expected index buffer to be already allocated");

                self.index_bufs[index_buf_idx].update_tasks.push(i);
            }
        }

        // Recalculate bounding boxes
        for data in &mut self.mesh_data {
            if !data.aabb_dirty {
                continue;
            }

            let (x, y, z) = calc_aabb(&data.vpos_vec);
            data.aabb_dirty = false;
            data.aabb_x = x;
            data.aabb_y = y;
            data.aabb_z = z;
        }

        // Update mesh buffers
        for vb in &mut self.vertex_bufs {
            if vb.update_tasks.is_empty() {
                continue;
            }
            gl::BindBuffer(gl::ARRAY_BUFFER, vb.buf_id);

            for &mesh_idx in &vb.update_tasks {
                let mesh = &mut self.mesh_data[mesh_idx];
                let vertex_buf_len = mesh.vertex_buf_len();

                let range_start = mesh.vertex_buf_alloc
                    .as_ref()
                    .map(|&(_, ref range)| range.start())
                    .expect("Expected vertex buffer to be already allocated");

                let map_base = (range_start * 9 * 4) as GLsizeiptr;
                let map_size = (vertex_buf_len * 9 * 4) as GLsizeiptr;

                let map_flags = if self.buggy_intel {
                    gl::MAP_WRITE_BIT
                } else {
                    gl::MAP_WRITE_BIT | gl::MAP_UNSYNCHRONIZED_BIT
                };
                let map = gl::MapBufferRange(gl::ARRAY_BUFFER, map_base, map_size, map_flags) as *mut f32;
                assert_ne!(map, ptr::null_mut());

                for i in 0..vertex_buf_len {
                    let vpos = mesh.vpos_vec.get(i).map(|x| *x).unwrap_or(Vector3::zero());
                    let vnorm = mesh.vnorm_vec.get(i).map(|x| *x).unwrap_or(Vector3::zero());
                    let vcolour = mesh.vcolour_vec.get(i).map(|x| *x).unwrap_or(Vector3::zero());
                    let base = map.offset(i as isize * 9);
                    *base.offset(0) = vpos.x;
                    *base.offset(1) = vpos.y;
                    *base.offset(2) = vpos.z;
                    *base.offset(3) = vnorm.x;
                    *base.offset(4) = vnorm.y;
                    *base.offset(5) = vnorm.z;
                    *base.offset(6) = vcolour.x;
                    *base.offset(7) = vcolour.y;
                    *base.offset(8) = vcolour.z;
                }

                gl::UnmapBuffer(gl::ARRAY_BUFFER);

                mesh.vertex_buf_dirty = false;
            }
            vb.update_tasks.clear();
        }

        for ib in &mut self.index_bufs {
            if ib.update_tasks.is_empty() {
                continue;
            }
            gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, ib.buf_id);

            for &mesh_idx in &ib.update_tasks {
                let mesh = &mut self.mesh_data[mesh_idx];
                let indices_buf_len = mesh.indices_vec.len();

                let range_start = mesh.index_buf_alloc
                    .as_ref()
                    .map(|&(_, ref range)| range.start())
                    .expect("Expected index buffer to be already allocated");

                let map_base = (range_start * 2) as GLsizeiptr;
                let map_size = (indices_buf_len * 2) as GLsizeiptr;

                let map_flags = if self.buggy_intel {
                    gl::MAP_WRITE_BIT
                } else {
                    gl::MAP_WRITE_BIT | gl::MAP_UNSYNCHRONIZED_BIT
                };
                let map = gl::MapBufferRange(gl::ELEMENT_ARRAY_BUFFER, map_base, map_size, map_flags) as *mut u16;
                assert_ne!(map, ptr::null_mut());

                ptr::copy_nonoverlapping(mesh.indices_vec.as_ptr(), map, indices_buf_len);

                gl::UnmapBuffer(gl::ELEMENT_ARRAY_BUFFER);

                mesh.index_buf_dirty = false;
            }
            ib.update_tasks.clear();
        }

        let fence = if !self.buggy_intel {
            let fence = gl::FenceSync(gl::SYNC_GPU_COMMANDS_COMPLETE, 0);
            if fence == ptr::null() {
                panic!("glFenceSync unexpectedly failed.");
            }
            fence
        } else {
            ptr::null()
        };

        self.fences[self.cur_fence] = Some(fence);
        self.cur_fence += 1;
        if self.cur_fence >= self.fences.len() {
            self.cur_fence = 0;
        }

        for i in 0..self.object_data.c.trans_data.len() {
            let trans_data = &self.object_data.c.trans_data[i];

            // Check whether we need to recalculate the local-to-world matrix.
            let mut opt_data = Some(trans_data);
            let mut recalc = false;
            while let Some(data) = opt_data {
                if !trans_data.ltw_valid.get() {
                    recalc = true;
                    break
                }
                opt_data = data.parent_idx
                    .map(|idx| &self.object_data.c.trans_data[idx]);
            }

            if recalc {
                let (world_pos, world_rot) = {
                    let parent_data = trans_data.parent_idx
                        .map(|idx| &self.object_data.c.trans_data[idx]);
                    self.local_to_world_pos_rot(parent_data, trans_data.pos, trans_data.rot)
                };

                let local_to_world = (Matrix4::from_translation(world_pos)
                    * Matrix4::from(world_rot)).into();
                trans_data.ltw_matrix.set(local_to_world);
            }
        }

        // All local-to-world matrices have been recomputed where necessary,
        // mark as valid going in to the next frame.
        for trans_data in &self.object_data.c.trans_data {
            trans_data.ltw_valid.set(true);
        }

        // Calculate camera matrices and positions
        let mut camera_data = Vec::new();

        std::mem::swap(&mut camera_data, &mut self.camera_data.c.data);
        for camera in &mut camera_data {
            let trans_idx = self.object_data.data_idx_checked(camera.parent)
                .expect("Camera not destroyed along with parent");
            let trans_data = &self.object_data.c.trans_data[trans_idx];
            let (cam_pos, cam_rot) = {
                let parent_data = trans_data.parent_idx
                    .map(|idx| &self.object_data.c.trans_data[idx]);
                self.local_to_world_pos_rot(parent_data, trans_data.pos, trans_data.rot)
            };

            let cam_perspective = cgmath::perspective(camera.fovy,
                                                      camera.aspect,
                                                      camera.near_clip,
                                                      camera.far_clip);

            let pv_matrix =
                cam_perspective *
                Matrix4::from(cam_rot.invert()) *
                Matrix4::from_translation(-cam_pos);

            camera.pos = cam_pos;
            camera.pv_matrix = pv_matrix;
        }
        std::mem::swap(&mut camera_data, &mut self.camera_data.c.data);

        for camera in &mut self.camera_data.c.data {
            camera.draw_cmds.clear();
        }

        // Generate draw commands
        for data in &self.mrenderer_data.c.data {
            if !data.enabled {
                continue;
            }

            let trans_idx = self.object_data.data_idx_checked(data.parent)
                .expect("MeshRenderer not destroyed along with parent");
            let trans_data = &self.object_data.c.trans_data[trans_idx];

            let mesh_idx = data.mesh
                .as_ref()
                .and_then(|x| x.idx.get());
            let mesh_idx = match mesh_idx {
                Some(mesh_idx) => mesh_idx,
                None => continue,
            };
            let mesh = &self.mesh_data[mesh_idx];

            let material_idx = data.material.map(|handle| self.material_data.data_idx_checked(handle));
            let material_idx = match material_idx {
                Some(Ok(material_idx)) => material_idx,
                _ => continue,
            };
            let material = &self.material_data.c.data[material_idx];

            let shader_idx = self.shader_data.data_idx_checked(material.shader);
            let shader_idx = match shader_idx {
                Ok(shader_idx) => shader_idx,
                Err(_) => continue,
            };

            let vertex_buf_idx = mesh.vertex_buf_alloc
                .as_ref()
                .map(|&(ref idx, _)| *idx)
                .expect("Expected vertex buffer to be already allocated");
            let index_buf_idx = mesh.index_buf_alloc
                .as_ref()
                .map(|&(ref idx, _)| *idx)
                .expect("Expected index buffer to be already allocated");

            for camera in &mut self.camera_data.c.data {
                if !camera.enabled {
                    continue;
                }

                if (camera.layers & data.layers) == 0 {
                    // This camera doesn't render any of the mesh renderer's
                    // layers
                    continue;
                }

                let obj_matrix = Matrix4::from(trans_data.ltw_matrix.get());
                // Need to transpose since our matrix is actually PVM instead of MVP
                let mvp = (camera.pv_matrix * obj_matrix).transpose();

                if intersects_frustum(mvp, mesh.aabb_x, mesh.aabb_y, mesh.aabb_z) {
                    let info = DrawInfo {
                        shader_idx,
                        material_idx,
                        vertex_buf_idx,
                        index_buf_idx,
                        trans_idx,
                        mesh_idx
                    };
                    camera.draw_cmds.push(info);
                }
            }
        }

        // Sort draw commands for batching
        for camera in &mut self.camera_data.c.data {
            camera.draw_cmds.sort_by(|a, b| {
                match a.shader_idx.cmp(&b.shader_idx) {
                    Ordering::Equal => (),
                    order => return order,
                }

                match a.material_idx.cmp(&b.material_idx) {
                    Ordering::Equal => (),
                    order => return order,
                }

                match a.vertex_buf_idx.cmp(&b.vertex_buf_idx) {
                    Ordering::Equal => (),
                    order => return order,
                }

                match a.index_buf_idx.cmp(&b.index_buf_idx) {
                    Ordering::Equal => (),
                    order => return order,
                }

                Ordering::Equal
            });
        }

        let mut camera_render_order = Vec::new();

        std::mem::swap(&mut self.camera_render_order, &mut camera_render_order);

        camera_render_order.clear();
        camera_render_order.extend(0..self.camera_data.c.data.len());

        // Sort cameras into the order we want to render them in
        camera_render_order.sort_by(|&a, &b| {
            let a_order = self.camera_data.c.data[a].order;
            let b_order = self.camera_data.c.data[b].order;
            a_order.cmp(&b_order)
        });

        std::mem::swap(&mut self.camera_render_order, &mut camera_render_order);

        let mut prev_shader_idx = None;
        let mut prev_material_idx = None;
        let mut prev_vertex_buf_idx = None;
        let mut prev_index_buf_idx = None;

        gl::Clear(gl::COLOR_BUFFER_BIT);
        // Execute draw commands
        for &camera_idx in &self.camera_render_order {
            let camera = &self.camera_data.c.data[camera_idx];
            let pv_matrix: [[f32; 4]; 4] = camera.pv_matrix.into();
            let cam_pos: [f32; 3] = camera.pos.into();

            gl::Clear(gl::DEPTH_BUFFER_BIT);
            for info in &camera.draw_cmds {
                if prev_shader_idx != Some(info.shader_idx) {
                    let shader = &self.shader_data.c.data[info.shader_idx];
                    gl::UseProgram(shader.program.program);
                    prev_shader_idx = Some(info.shader_idx);
                }
                if prev_material_idx != Some(info.material_idx) {
                    prev_material_idx = Some(info.material_idx);
                }
                if prev_vertex_buf_idx != Some(info.vertex_buf_idx) {
                    let vao_id = self.vertex_bufs[info.vertex_buf_idx].vao_id;
                    gl::BindVertexArray(vao_id);
                    prev_vertex_buf_idx = Some(info.vertex_buf_idx);
                }
                if prev_index_buf_idx != Some(info.index_buf_idx) {
                    let index_buf_id = self.index_bufs[info.index_buf_idx].buf_id;
                    gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER, index_buf_id);
                    prev_index_buf_idx = Some(info.index_buf_idx);
                }

                let shader = &self.shader_data.c.data[info.shader_idx];
                let trans_data = &self.object_data.c.trans_data[info.trans_idx];
                let mesh = &self.mesh_data[info.mesh_idx];
                let material = &self.material_data.c.data[info.material_idx];

                let vertex_base = mesh.vertex_buf_alloc
                    .as_ref()
                    .map(|&(_, ref range)| range.start())
                    .expect("Expected vertex buffer to be already allocated");
                let index_base = mesh.index_buf_alloc
                    .as_ref()
                    .map(|&(_, ref range)| range.start())
                    .expect("Expected index buffer to be already allocated");

                let colour = [material.colour.0, material.colour.1, material.colour.2];

                gl::UniformMatrix4fv(shader.obj_matrix_uniform, 1, gl::FALSE, trans_data.ltw_matrix.get().as_ptr() as *const _);
                gl::Uniform3fv(shader.cam_pos_uniform, 1, cam_pos.as_ptr());
                gl::UniformMatrix4fv(shader.cam_matrix_uniform, 1, gl::FALSE, pv_matrix.as_ptr() as *const _);
                gl::Uniform3fv(shader.colour_uniform, 1, colour.as_ptr());

                for &(location, ref value) in &material.uniform_values {
                    let value = match *value {
                        Some(ref value) => value,
                        None => continue,
                    };
                    match *value {
                        UniformValue::Mat2(x) => {
                            gl::UniformMatrix2fv(location, 1, gl::FALSE, x.as_ptr() as *const _);
                        }
                        UniformValue::Mat3(x) => {
                            gl::UniformMatrix3fv(location, 1, gl::FALSE, x.as_ptr() as *const _);
                        }
                        UniformValue::Mat4(x) => {
                            gl::UniformMatrix4fv(location, 1, gl::FALSE, x.as_ptr() as *const _);
                        }
                        UniformValue::Float(x) => {
                            gl::Uniform1fv(location, 1, &x);
                        }
                        UniformValue::Vec2(x) => {
                            gl::Uniform2fv(location, 1, x.as_ptr());
                        }
                        UniformValue::Vec3(x) => {
                            gl::Uniform3fv(location, 1, x.as_ptr());
                        }
                        UniformValue::Vec4(x) => {
                            gl::Uniform4fv(location, 1, x.as_ptr());
                        }
                        UniformValue::Int(x) => {
                            gl::Uniform1iv(location, 1, &x);
                        }
                        UniformValue::IntVec2(x) => {
                            gl::Uniform2iv(location, 1, x.as_ptr());
                        }
                        UniformValue::IntVec3(x) => {
                            gl::Uniform3iv(location, 1, x.as_ptr());
                        }
                        UniformValue::IntVec4(x) => {
                            gl::Uniform4iv(location, 1, x.as_ptr());
                        }
                        UniformValue::UInt(x) => {
                            gl::Uniform1uiv(location, 1, &x);
                        }
                        UniformValue::UIntVec2(x) => {
                            gl::Uniform2uiv(location, 1, x.as_ptr());
                        }
                        UniformValue::UIntVec3(x) => {
                            gl::Uniform3uiv(location, 1, x.as_ptr());
                        }
                        UniformValue::UIntVec4(x) => {
                            gl::Uniform4uiv(location, 1, x.as_ptr());
                        }
                        UniformValue::Cubemap(x) => {
                            let idx = self.cubemap_data.data_idx_checked(x)
                                .expect("Cubemap in use while destroyed");
                            let texture_id = self.cubemap_data.c.data[idx].texture_id;
                            // FIXME: massive hack
                            gl::ActiveTexture(gl::TEXTURE0 + 0);
                            gl::BindTexture(gl::TEXTURE_CUBE_MAP, texture_id);
                            gl::Uniform1i(location, 0);
                        }
                    }
                }

                // TODO: exploit multi-draw where possible

                gl::DrawElementsBaseVertex(gl::TRIANGLES,
                                           mesh.indices_vec.len() as GLsizei,
                                           gl::UNSIGNED_SHORT,
                                           (index_base * 2) as *const GLvoid,
                                           vertex_base as GLint);
            }
        }

        self.window.as_ref().unwrap().gl_swap_window();

        // Event loop
        for event in self.event_pump.as_mut().unwrap().poll_iter() {
            match event {
                Event::Quit {..} => return false,
                Event::AppTerminating {..} => return false,
                _ => (),
            }
        };

        true
    }

    fn local_to_world_pos_rot(&self,
                              parent_data: Option<&TransformData>,
                              input_pos: Vector3<f32>,
                              input_rot: Quaternion<f32>)
                              -> (Vector3<f32>, Quaternion<f32>) {
        let mut tmp_vec = unsafe { &mut *self.tmp_vec.get() };
        tmp_vec.clear();
        let mut opt_data = parent_data;
        while let Some(data) = opt_data {
            tmp_vec.push((data.pos, data.rot));
            opt_data = data.parent_idx
                .map(|idx| unsafe { self.object_data.c.trans_data.get_unchecked(idx) });
        }
        let mut world_pos = Vector3::new(0.0, 0.0, 0.0);
        let mut c = Vector3::new(0.0, 0.0, 0.0);
        let mut world_rot = Quaternion::one();
        for &(local_pos, local_rot) in tmp_vec.iter() {
            let add = world_rot * local_pos;
            let y = add - c;
            let t = world_pos + y;
            c = (t - world_pos) - y;
            world_pos = t;
            world_rot = world_rot * local_rot;
        }
        let add = world_rot * input_pos;
        let y = add - c;
        let t = world_pos + y;
        world_pos = t;
        world_rot = world_rot * input_rot;
        (world_pos, world_rot)
    }

    fn local_to_world_pos(&self, parent_data: Option<&TransformData>, input_pos: Vector3<f32>)
                          -> Vector3<f32> {
        let (world_pos, _) = self.local_to_world_pos_rot(parent_data, input_pos, Quaternion::one());
        world_pos
    }

    fn local_to_world_rot(&self, parent_data: Option<&TransformData>, input_rot: Quaternion<f32>)
                          -> Quaternion<f32> {
        let mut opt_data = parent_data;
        let mut world_rot = input_rot;
        while let Some(data) = opt_data {
            world_rot = data.rot * world_rot;
            opt_data = data.parent_idx
                .map(|idx| unsafe { self.object_data.c.trans_data.get_unchecked(idx) });
        }
        world_rot
    }

    fn world_to_local_pos_rot(&self,
                              parent_data: Option<&TransformData>,
                              input_pos: Vector3<f32>,
                              input_rot: Quaternion<f32>)
                              -> (Vector3<f32>, Quaternion<f32>) {
        let (world_pos, world_rot) =
            self.local_to_world_pos_rot(parent_data, Vector3::zero(), Quaternion::one());
        // Cheat here by using double precision until I can figure out if there
        // is another way to improve precision.
        let input_rot = Quaternion::<f64>::new(input_rot.s as f64,
                                               input_rot.v.x as f64,
                                               input_rot.v.y as f64,
                                               input_rot.v.z as f64);
        let world_rot = Quaternion::<f64>::new(world_rot.s as f64,
                                               world_rot.v.x as f64,
                                               world_rot.v.y as f64,
                                               world_rot.v.z as f64);
        let input_pos = Vector3::<f64>::new(input_pos.x as f64,
                                            input_pos.y as f64,
                                            input_pos.z as f64);
        let world_pos = Vector3::<f64>::new(world_pos.x as f64,
                                            world_pos.y as f64,
                                            world_pos.z as f64);
        let inv_world_rot = world_rot.invert();
        let local_pos = inv_world_rot * (input_pos - world_pos);
        let local_rot = inv_world_rot * input_rot;
        let local_rot = Quaternion::<f32>::new(local_rot.s as f32,
                                               local_rot.v.x as f32,
                                               local_rot.v.y as f32,
                                               local_rot.v.z as f32);
        let local_pos = Vector3::<f32>::new(local_pos.x as f32,
                                            local_pos.y as f32,
                                            local_pos.z as f32);
        (local_pos, local_rot)
    }

    fn world_to_local_pos(&self, parent_data: Option<&TransformData>, input_pos: Vector3<f32>)
                          -> Vector3<f32> {
        let (local_pos, _) = self.world_to_local_pos_rot(parent_data, input_pos, Quaternion::one());
        local_pos
    }

    fn world_to_local_rot(&self, parent_data: Option<&TransformData>, input_rot: Quaternion<f32>)
                          -> Quaternion<f32> {
        let world_rot = self.local_to_world_rot(parent_data, Quaternion::one());
        let inv_world_rot = world_rot.invert();
        let local_rot = inv_world_rot * input_rot;
        local_rot
    }

    pub fn keyboard_state(&self) -> KeyboardState {
        KeyboardState::new(self.event_pump.as_ref().unwrap())
    }

    pub fn mouse_state(&self) -> MouseState {
        MouseState::new(self.event_pump.as_ref().unwrap())
    }

    pub fn set_object_parent(&mut self, object: Handle<Object>, parent: Option<Handle<Object>>) {
        let obj_idx = self.object_data.data_idx_checked(object);
        let parent_idx = parent.map(|o| self.object_data.data_idx_checked(o));

        let obj_idx = match obj_idx {
            Ok(obj_idx) => obj_idx,
            Err(_) => {
                // The object doesn't have a valid handle, stop.
                // TODO: maybe be less forgiving and just panic!()?
                return;
            }
        };

        let parent_idx = match parent_idx {
            Some(Ok(parent_idx)) => Some(parent_idx),
            Some(Err(_)) => {
                // The parent doesn't have a valid handle, stop.
                // TODO: maybe be less forgiving and just panic!()?
                return;
            }
            None => None,
        };

        if parent_idx == Some(obj_idx) {
            // Disallow parenting to self
            // TODO: maybe be less forgiving and just panic!()?
            return;
        }

        if self.check_nop(obj_idx, parent_idx) {
            // This parenting would be a no-op.
            return;
        }

        if !self.check_parenting(obj_idx, parent_idx) {
            // Performing this parenting is not allowed.
            // TODO: maybe be less forgiving and just panic!()?
            return;
        }

        self.unparent(obj_idx);
        self.reparent(obj_idx, parent_idx);

        // Reparenting can change the local-to-world matrix in unpredictable
        // ways.
        let trans_data = &self.object_data.c.trans_data[obj_idx];
        trans_data.ltw_valid.set(false);
    }

    fn check_nop(&self, object_idx: usize, parent_idx: Option<usize>) -> bool {
        let obj_trans_data = &self.object_data.c.trans_data[object_idx];
        obj_trans_data.parent_idx == parent_idx
    }

    fn check_parenting(&self, object_idx: usize, parent_idx: Option<usize>) -> bool {
        let parent_obj_data = parent_idx.map(|idx| &self.object_data.c.obj_data[idx]);
        let parent_trans_data = parent_idx.map(|idx| &self.object_data.c.trans_data[idx]);
        if parent_obj_data.as_ref().map_or(false, |x| x.marked) {
            // Can't parent to something marked for destruction
            return false
        }

        let mut opt_idx = parent_trans_data.as_ref()
            .and_then(|parent_data| parent_data.parent_idx);
        while let Some(idx) = opt_idx {
            if idx == object_idx {
                // Performing this parenting would create a loop.
                return false
            }
            opt_idx = self.object_data.c.trans_data[idx].parent_idx;
        }

        true
    }

    fn unparent(&mut self, object_idx: usize) {
        let handle = self.object_data.handle(object_idx);
        {
            let old_parent_idx = self.object_data.c.trans_data[object_idx].parent_idx;
            let old_parent_trans_data =
                old_parent_idx.map(|i| &mut self.object_data.c.trans_data[i]);

            if let Some(old_parent_trans_data) = old_parent_trans_data {
                old_parent_trans_data.children.iter()
                    .position(|&handle2| handle == handle2)
                    .map(|e| old_parent_trans_data.children.remove(e))
                    .expect("parent should contain child index");
            }
        }

        let trans_data = &mut self.object_data.c.trans_data[object_idx];
        trans_data.parent_idx = None;
    }

    fn reparent(&mut self, object_idx: usize, parent_idx: Option<usize>) {
        let handle = self.object_data.handle(object_idx);
        // Assume we currently have no parent, if parent_idx is `None`, we can
        // return early.
        let parent_idx = match parent_idx {
            Some(idx) => idx,
            None => return,
        };

        {
            let parent_trans_data = &mut self.object_data.c.trans_data[parent_idx];
            parent_trans_data.children.push(handle);
        }

        let trans_data = &mut self.object_data.c.trans_data[object_idx];
        trans_data.parent_idx = Some(parent_idx);
    }

    fn fixup_hierarchy(object_data: &mut ObjectManager<ObjectContainer>, old_idx: usize, new_idx: usize) {
        let mut tmp_children = Vec::new();
        std::mem::swap(&mut tmp_children, &mut object_data.c.trans_data[old_idx].children);

        // Update our children's reference to us
        for &child in &tmp_children {
            let child_idx = object_data.data_idx_checked(child)
                .expect("Destroyed object found in hierarchy");
            let child_data = &mut object_data.c.trans_data[child_idx];
            child_data.parent_idx = Some(new_idx);
        }
        std::mem::swap(&mut tmp_children, &mut object_data.c.trans_data[old_idx].children);
    }

    fn cleanup_destroyed_objects(&mut self) {
        for &handle in &self.destroyed_objects {
            // Remove destroyed objects at the back of the list
            while self.object_data.c.obj_data.last().map_or(false, |x| x.marked) {
                let old_idx = self.object_data.c.trans_data.len() - 1;

                // No need to fix up hierarchy since this doesn't change the
                // index of any non-marked objects.
                self.object_data.remove_idx(old_idx);
            }

            let idx = match self.object_data.data_idx_checked(handle) {
                Ok(idx) => idx,
                Err(_) => continue, // This object has already been removed in the loop above.
            };

            let swapped_idx = self.object_data.c.trans_data.len() - 1;
            Self::fixup_hierarchy(&mut self.object_data, swapped_idx, idx);
            self.object_data.remove(handle)
                .expect("Destroyed object found in hierarchy");
        }
        self.destroyed_objects.clear();
    }

    unsafe fn cleanup_destroyed<T, F, G>(items: &mut Vec<T>,
                                         destroyed_items: &mut Vec<usize>,
                                         is_destroyed: F,
                                         mut set_idx: G)
        where F: Fn(&T) -> bool, G: FnMut(&T, Option<usize>) {
        for &idx in destroyed_items.iter() {
            // Remove destroyed objects at the back of the list
            while items.last().map_or(false, |x| is_destroyed(x)) {
                let removed = items.pop().unwrap();
                set_idx(&removed, None);
            }
            if idx >= items.len() {
                continue
            }
            let removed = items.swap_remove(idx);
            set_idx(&removed, None);
            let swapped = items.get_unchecked(idx);
            set_idx(swapped, Some(idx));
        }
        destroyed_items.clear();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cgmath::{Deg, Euler, Quaternion, Vector3};
    use num::{Zero, One};

    /// Tests integrity of the Scene hierarchy after destroying objects.
    #[test]
    fn test_hierarchy_integrity() {
        let mut scene = Scene::new_headless();

        // Set up the hierarchy
        let root_obj = scene.create_object();

        let child1 = scene.create_object();
        scene.set_object_parent(child1, Some(root_obj));

        let child2 = scene.create_object();
        scene.set_object_parent(child2, Some(root_obj));

        let child11 = scene.create_object();
        scene.set_object_parent(child11, Some(child1));
        let child12 = scene.create_object();
        scene.set_object_parent(child12, Some(child1));
        let child13 = scene.create_object();
        scene.set_object_parent(child13, Some(child1));

        let child21 = scene.create_object();
        scene.set_object_parent(child21, Some(child2));
        let child22 = scene.create_object();
        scene.set_object_parent(child22, Some(child2));
        let child23 = scene.create_object();
        scene.set_object_parent(child23, Some(child2));

        scene.do_frame();

        // Verify it is what we expect
        assert_eq!(root_obj.children(&scene)
                   .map(|x| x[0] == child1).ok(),
                   Some(true));
        assert_eq!(root_obj.children(&scene)
                   .map(|x| x[1] == child2).ok(),
                   Some(true));

        assert_eq!(child1.children(&scene)
                   .map(|x| x[0] == child11).ok(),
                   Some(true));
        assert_eq!(child1.children(&scene)
                   .map(|x| x[1] == child12).ok(),
                   Some(true));
        assert_eq!(child1.children(&scene)
                   .map(|x| x[2] == child13).ok(),
                   Some(true));

        assert_eq!(child2.children(&scene)
                   .map(|x| x[0] == child21).ok(),
                   Some(true));
        assert_eq!(child2.children(&scene)
                   .map(|x| x[1] == child22).ok(),
                   Some(true));
        assert_eq!(child2.children(&scene)
                   .map(|x| x[2] == child23).ok(),
                   Some(true));

        // Destroy the objects and run a frame so the hierarchy is changed
        scene.destroy_object(child2);
        scene.destroy_object(child12);
        scene.do_frame();

        assert_eq!(root_obj.children(&scene)
                   .map(|x| x.len()).ok(), Some(1));
        assert_eq!(root_obj.children(&scene)
                   .map(|x| x[0] == child1).ok(),
                   Some(true));

        assert_eq!(child1.children(&scene)
                   .map(|x| x.len()).ok(), Some(2));
        assert_eq!(child1.children(&scene)
                   .map(|x| x[0] == child11).ok(),
                   Some(true));
        assert_eq!(child1.children(&scene)
                   .map(|x| x[1] == child13).ok(),
                   Some(true));

        assert!(child2.children(&scene).is_err());
    }

    /// Test to verify we cannot create cycles in the hierarchy.
    #[test]
    fn test_no_cycles() {
        let mut scene = Scene::new_headless();

        // Set up the hierarchy
        let root_obj = scene.create_object();

        let child_obj = scene.create_object();

        scene.set_object_parent(child_obj, Some(root_obj));
        // This should fail and do nothing
        scene.set_object_parent(root_obj, Some(child_obj));

        assert_eq!(root_obj.children(&scene)
                   .map(|x| x.len()).ok(), Some(1));
        assert_eq!(root_obj.children(&scene)
                   .map(|x| x[0] == child_obj).ok(),
                   Some(true));

        assert_eq!(child_obj.children(&scene)
                   .map(|x| x.len()).ok(), Some(0));

        // Set up the hierarchy
        let obj1 = scene.create_object();
        let obj2 = scene.create_object();
        let obj3 = scene.create_object();
        let obj4 = scene.create_object();
        let obj5 = scene.create_object();
        scene.set_object_parent(obj2, Some(obj1));
        scene.set_object_parent(obj3, Some(obj2));
        scene.set_object_parent(obj4, Some(obj3));
        scene.set_object_parent(obj5, Some(obj4));
        // This should fail and do nothing
        scene.set_object_parent(obj1, Some(obj5));

        assert_eq!(obj5.children(&scene)
                   .map(|x| x.len()).ok(), Some(0));
    }

    fn angles(x: f32, y: f32, z: f32) -> Quaternion<f32> {
        Quaternion::from(Euler { x: Deg(x), y: Deg(y), z: Deg(z) })
    }

    /// Tests objects are transformed correctly
    #[test]
    fn test_obj_transforms() {
        let mut scene = Scene::new_headless();

        let obj1 = scene.create_object();
        let obj2 = scene.create_object();
        let obj3 = scene.create_object();
        let obj4 = scene.create_object();
        let obj5 = scene.create_object();
        scene.set_object_parent(obj2, Some(obj1));
        scene.set_object_parent(obj3, Some(obj2));
        scene.set_object_parent(obj4, Some(obj3));
        scene.set_object_parent(obj5, Some(obj4));

        obj1.set_local_pos(&mut scene, Vector3::new(10.0, 0.0, 0.0)).unwrap();
        obj1.set_local_rot(&mut scene, angles(0.0, 0.0, 45.0)).unwrap();

        obj2.set_local_pos(&mut scene, Vector3::new(10.0, 0.0, 0.0)).unwrap();
        obj2.set_local_rot(&mut scene, angles(0.0, 0.0, 45.0)).unwrap();

        obj3.set_local_pos(&mut scene, Vector3::new(10.0, 0.0, 0.0)).unwrap();
        obj3.set_local_rot(&mut scene, angles(0.0, 0.0, 45.0)).unwrap();

        obj4.set_local_pos(&mut scene, Vector3::new(10.0, 0.0, 0.0)).unwrap();
        obj4.set_local_rot(&mut scene, angles(0.0, 0.0, 45.0)).unwrap();

        obj5.set_local_pos(&mut scene, Vector3::new(10.0, 0.0, 0.0)).unwrap();
        obj5.set_local_rot(&mut scene, angles(0.0, 0.0, 45.0)).unwrap();

        obj4.set_world_pos(&mut scene, Vector3::zero()).unwrap();
        obj4.set_world_rot(&mut scene, Quaternion::one()).unwrap();

        let tmp = obj1.world_pos(&scene).unwrap();
        assert_relative_eq!(tmp.x, 10.0);
        assert_relative_eq!(tmp.y, 0.0);
        assert_relative_eq!(tmp.z, 0.0);

        let tmp = obj2.world_pos(&scene).unwrap();
        assert_relative_eq!(tmp.x, 17.071067812);
        assert_relative_eq!(tmp.y, 7.071067812);
        assert_relative_eq!(tmp.z, 0.0);

        let tmp = obj3.world_pos(&scene).unwrap();
        assert_relative_eq!(tmp.x, 17.071067812);
        assert_relative_eq!(tmp.y, 17.071067812);
        assert_relative_eq!(tmp.z, 0.0);

        let tmp = obj4.world_pos(&scene).unwrap();
        assert_relative_eq!(tmp.x, 0.0);
        assert_relative_eq!(tmp.y, 0.0);
        assert_relative_eq!(tmp.z, 0.0);

        let tmp = obj5.world_pos(&scene).unwrap();
        assert_relative_eq!(tmp.x, 10.0, max_relative = 1.05);
        assert_relative_eq!(tmp.y, 0.0, max_relative = 1.05);
        assert_relative_eq!(tmp.z, 0.0);
    }

    /// Tests objects world rotations are calculated correctly
    #[test]
    fn test_obj_rotation() {
                let mut scene = Scene::new_headless();

        let obj1 = scene.create_object();
        let obj2 = scene.create_object();
        scene.set_object_parent(obj2, Some(obj1));

        obj1.set_local_rot(&mut scene, angles(-90.0, 0.0, 0.0)).unwrap();
        obj2.set_local_rot(&mut scene, angles(0.0, 0.0, -90.0)).unwrap();

        let tmp = obj2.world_rot(&scene).unwrap();
        let up = Vector3::new(0.0, 1.0, 0.0);
        let up_world = tmp * up;
        assert_relative_eq!(up_world.x, 1.0);
        assert_relative_eq!(up_world.y, 0.0);
        assert_relative_eq!(up_world.z, 0.0);
    }

    // FIXME: reimplement test `test_add_component`, add test for new behaviours
}
