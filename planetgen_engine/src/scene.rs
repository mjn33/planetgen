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
use std::cell::{Cell, RefCell, UnsafeCell};
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::ffi::CStr;
use std::ptr;
use std::rc::Rc;
use std::time::Instant;
use traits::Component;

fn post_add<T: Copy + std::ops::Add<Output=T>>(a: &mut T, b: T) -> T {
    let c = *a;
    *a = *a + b;
    c
}

fn calc_aabb(vertices: &[Vector3<f32>]) -> (Vector3<f32>, Vector3<f32>) {
    if vertices.len() == 0 {
        return (Vector3::zero(), Vector3::zero());
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

    (Vector3::new(min_x, min_y, min_z), Vector3::new(max_x, max_y, max_z))

}

fn aabb_points(min: Vector3<f32>, max: Vector3<f32>) -> [Vector3<f32>; 8] {
    [Vector3::new(min.x, min.y, min.z),
     Vector3::new(min.x, min.y, max.z),
     Vector3::new(min.x, max.y, min.z),
     Vector3::new(min.x, max.y, max.z),
     Vector3::new(max.x, min.y, min.z),
     Vector3::new(max.x, min.y, max.z),
     Vector3::new(max.x, max.y, min.z),
     Vector3::new(max.x, max.y, max.z)]
}

fn points_in_frustum(mvp: Matrix4<f32>, points: &[Vector3<f32>]) -> bool {
    let xn_plane = mvp.x + mvp.w;
    let xp_plane = -mvp.x + mvp.w;
    let yn_plane = mvp.y + mvp.w;
    let yp_plane = -mvp.y + mvp.w;
    let zn_plane = mvp.z + mvp.w;
    let zp_plane = -mvp.z + mvp.w;

    let planes = [xn_plane, xp_plane, yn_plane, yp_plane, zn_plane, zp_plane];

    for plane in planes.iter() {
        let mut out = true;
        for p in points {
            if plane.x * p.x + plane.y * p.y + plane.z * p.z + plane.w > 0.0 {
                out = false;
                break;
            }
        }

        if out {
            return false;
        }
    }

    true
}

struct CameraData {
    /// Reference to the camera object.
    camera: Rc<Camera>,
    /// Reference to the object we are a component of.
    parent: Rc<Object>,
    /// True when this camera should be used for rendering.
    enabled: bool,
    /// True when the camera has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    /// An integer describing the order in which this camera is to be
    /// renderered, higher values mean later.
    order: i32,
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

/// A handle to a camera object for a scene.
pub struct Camera {
    idx: Cell<Option<usize>>
}

impl Camera {
    pub fn set_fovy(&self, scene: &mut Scene, fovy: f32) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                let data = &mut scene.camera_data[i];
                data.fovy = Deg(fovy);
            })
    }

    pub fn set_near_clip(&self, scene: &mut Scene, near_clip: f32) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                let data = &mut scene.camera_data[i];
                data.near_clip = near_clip;
            })
    }

    pub fn set_far_clip(&self, scene: &mut Scene, far_clip: f32) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                let data = &mut scene.camera_data[i];
                data.far_clip = far_clip;
            })
    }

    pub fn set_order(&self, scene: &mut Scene, order: i32) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                let data = &mut scene.camera_data[i];
                data.order = order;
            })
    }

    pub fn order(&self, scene: &Scene) -> Result<i32> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                let data = &scene.camera_data[i];
                data.order
            })
    }

    pub fn is_valid(&self) -> bool {
        self.idx.get().is_some()
    }
}

impl Component for Camera {
    fn init(scene: &mut Scene, object: &Object) -> Result<Rc<Camera>> {
        scene.create_camera(object)
    }

    fn marked(&self, scene: &Scene) -> Result<bool> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                scene.camera_data[i].marked
            })
    }

    fn destroy(&self, scene: &mut Scene) {
        scene.destroy_camera(self);
    }
}

impl<T> Component for RefCell<T> where T: BehaviourMessages + 'static {
    fn init(scene: &mut Scene, object: &Object) -> Result<Rc<RefCell<T>>> {
        scene.create_behaviour::<T>(object)
    }

    fn marked(&self, scene: &Scene) -> Result<bool> {
        // FIXME: borrow() could fail, how should this be handled?
        self.borrow().behaviour().idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                scene.behaviour_data[i].marked
            })
    }

    fn destroy(&self, scene: &mut Scene) {
        // FIXME: borrow() could fail, how should this be handled?
        scene.destroy_behaviour2(&*self.borrow());
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
    aabb_min: Vector3<f32>,
    aabb_max: Vector3<f32>,
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

struct DrawInfo {
    shader_idx: usize,
    material_idx: usize,
    vertex_buf_idx: usize,
    index_buf_idx: usize,
    trans_idx: usize,
    mesh_idx: usize,
}

struct ShaderData {
    /// Reference to the shader object.
    object: Rc<Shader>,
    /// True when the shader has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    program: Program,
    obj_matrix_uniform: GLint,
    cam_pos_uniform: GLint,
    cam_matrix_uniform: GLint,
    colour_uniform: GLint,
}

pub struct Shader {
    idx: Cell<Option<usize>>
}

impl Shader {
    pub fn is_valid(&self) -> bool {
        self.idx.get().is_some()
    }
}

pub enum UniformValue {
    Int(i32),
    UnsignedInt(u32),
    Float(f32),
    Mat2([[f32; 2]; 2]),
    Mat3([[f32; 3]; 3]),
    Mat4([[f32; 4]; 4]),
    Vec2([f32; 2]),
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    IntVec2([i32; 2]),
    IntVec3([i32; 3]),
    IntVec4([i32; 4]),
    UIntVec2([u32; 2]),
    UIntVec3([u32; 3]),
    UIntVec4([u32; 4]),
    Bool(bool),
    BoolVec2([bool; 2]),
    BoolVec3([bool; 3]),
    BoolVec4([bool; 4]),
    Double(f64),
    DoubleVec2([f64; 2]),
    DoubleVec3([f64; 3]),
    DoubleVec4([f64; 4]),
    DoubleMat2([[f64; 2]; 2]),
    DoubleMat3([[f64; 3]; 3]),
    DoubleMat4([[f64; 4]; 4]),
    Int64(i64),
    Int64Vec2([i64; 2]),
    Int64Vec3([i64; 3]),
    Int64Vec4([i64; 4]),
    UInt64(u64),
    UInt64Vec2([u64; 2]),
    UInt64Vec3([u64; 3]),
    UInt64Vec4([u64; 4])
}

struct MaterialData {
    /// Reference to the material object.
    object: Rc<Material>,
    /// True when the material has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    uniforms: HashMap<String, UnsafeCell<Option<UniformValue>>>,
    shader: Rc<Shader>,
    colour: (f32, f32, f32)
}

pub struct Material {
    idx: Cell<Option<usize>>
}

impl Material {
    pub fn set_colour(&self, scene: &mut Scene, colour: (f32, f32, f32)) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe { scene.material_data.get_unchecked_mut(i).colour = colour; })
    }

    pub fn colour(&self, scene: &Scene) -> Result<(f32, f32, f32)> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe { scene.material_data.get_unchecked(i).colour })
    }

    pub fn set_uniform(&self, scene: &mut Scene, name: &str, v: UniformValue) -> Result<()> {
        let uniforms = try!(self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe { &mut scene.material_data.get_unchecked_mut(i).uniforms }));

        match uniforms.get(name) {
            Some(entry) => {
                unsafe {
                    *entry.get() = Some(v);
                }
                //entry.set_value(v);
                Ok(())
            },
            None => Err(Error::BadUniformName)
        }
    }

    pub fn is_valid(&self) -> bool {
        self.idx.get().is_some()
    }
}

struct MeshRendererData {
    /// Reference to the mesh renderer object.
    object: Rc<MeshRenderer>,
    /// Reference to the object we are a component of.
    parent: Rc<Object>,
    /// True when the mesh renderer has been marked for destruction at the end
    /// of the frame.
    marked: bool,
    enabled: bool,
    mesh: Option<Rc<Mesh>>,
    material: Option<Rc<Material>>,
}

pub struct MeshRenderer {
    idx: Cell<Option<usize>>,
}

impl MeshRenderer {
    pub fn set_enabled(&self, scene: &mut Scene, enabled: bool) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| scene.mrenderer_data[i].enabled = enabled)
    }

    pub fn set_mesh(&self, scene: &mut Scene, mesh: Option<Rc<Mesh>>) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| scene.mrenderer_data[i].mesh = mesh)
    }

    pub fn set_material(&self, scene: &mut Scene, material: Option<Rc<Material>>) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| scene.mrenderer_data[i].material = material)
    }

    pub fn enabled(&self, scene: &mut Scene) -> Result<bool> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| scene.mrenderer_data[i].enabled)
    }

    pub fn mesh<'a>(&self, scene: &'a mut Scene) -> Result<Option<&'a Rc<Mesh>>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(move |i| scene.mrenderer_data[i].mesh.as_ref())
    }

    pub fn material<'a>(&self, scene: &'a mut Scene) -> Result<Option<&'a Rc<Material>>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(move |i| scene.mrenderer_data[i].material.as_ref())
    }

    pub fn is_valid(&self) -> bool {
        self.idx.get().is_some()
    }
}

impl Component for MeshRenderer {
    fn init(scene: &mut Scene, object: &Object) -> Result<Rc<MeshRenderer>> {
        scene.create_mrenderer(object)
    }

    fn marked(&self, scene: &Scene) -> Result<bool> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| {
                scene.mrenderer_data[i].marked
            })
    }

    fn destroy(&self, scene: &mut Scene) {
        scene.destroy_mrenderer(self);
    }
}

pub trait BehaviourMessages {
    fn create(behaviour: Behaviour) -> Self where Self: Sized;

    fn start(&mut self, scene: &mut Scene);

    fn update(&mut self, scene: &mut Scene);

    fn destroy(&mut self, scene: &mut Scene);

    fn behaviour(&self) -> &Behaviour;
}

trait AnyBehaviour: Any {
    fn as_any(&self) -> &Any;
    /// Same as `Any::get_type_id` but stable.
    fn type_id(&self) -> TypeId;
    fn borrow(&self) -> std::cell::Ref<BehaviourMessages>;
    fn borrow_mut(&self) -> std::cell::RefMut<BehaviourMessages>;
}

trait AnyComponent: Any + Component {
    fn as_any(&self) -> &Any;
    fn type_id(&self) -> TypeId;
}

impl<T: Component + 'static> AnyComponent for T {
    fn as_any(&self) -> &Any {
        self
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }
}

impl<T: BehaviourMessages + 'static> AnyBehaviour for RefCell<T> {
    fn as_any(&self) -> &Any {
        self
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn borrow(&self) -> std::cell::Ref<BehaviourMessages> {
        RefCell::borrow(self)
    }

    fn borrow_mut(&self) -> std::cell::RefMut<BehaviourMessages> {
        RefCell::borrow_mut(self)
    }
}

struct BehaviourData {
    /// Reference to the behaviour implementation.
    behaviour: Rc<AnyBehaviour>,
    /// Reference to the object we are a component of.
    parent: Rc<Object>,
    /// True when the object has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    /// True if the object has been newly created this current frame.
    is_new: bool
}

pub struct Behaviour {
    idx: Cell<Option<usize>>,
}

impl Behaviour {
    pub fn object<'a>(&self, scene: &'a Scene) -> Result<&'a Rc<Object>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                &scene.behaviour_data.get_unchecked(i).parent
            })
    }

    pub fn is_valid(&self) -> bool {
        self.idx.get().is_some()
    }
}

struct ObjectData {
    /// Reference to the object.
    object: Rc<Object>,
    /// The components attached to this object.
    components: HashMap<TypeId, Rc<AnyComponent>>,
    /// True when the object has been marked for destruction at the end of the
    /// frame.
    marked: bool,
}

struct TransformData {
    /// Local rotation.
    rot: Quaternion<f32>,
    /// Local position.
    pos: Vector3<f32>,
    /// List of indices of the children of this object.
    children: Vec<usize>,
    /// Index of this object's parent.
    parent_idx: Option<usize>,
    /// Cached local-to-world matrix for this transform.
    ltw_matrix: Cell<[[f32; 4]; 4]>,
    /// False if our local rotation / position has been changed, or we've been
    /// reparented this frame. False will cause this transform and our child
    /// transforms to have their local-to-world matrix recomputed.
    ltw_valid: Cell<bool>,
}

pub struct Object {
    idx: Cell<Option<usize>>
}

impl Object {
    pub fn num_children(&self, scene: &Scene) -> Result<usize> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe { scene.transform_data.get_unchecked(i).children.len() })
    }

    pub fn get_child<'a>(&self, scene: &'a Scene, n: usize) -> Result<&'a Rc<Object>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .and_then(|i| unsafe {
                scene.transform_data.get_unchecked(i).children.get(n)
                    .ok_or(Error::BadChildIdx)
            })
            .map(|&i| unsafe {
                &scene.object_data.get_unchecked(i).object
            })
    }

    pub fn set_local_pos(&self, scene: &mut Scene, pos: Vector3<f32>) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                let data = scene.transform_data.get_unchecked_mut(i);
                data.pos = pos;
                data.ltw_valid.set(false);
            })
    }

    pub fn local_pos(&self, scene: &Scene) -> Result<Vector3<f32>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                scene.transform_data.get_unchecked(i).pos
            })
    }

    pub fn set_local_rot(&self, scene: &mut Scene, rot: Quaternion<f32>) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                let data = scene.transform_data.get_unchecked_mut(i);
                data.rot = rot;
                data.ltw_valid.set(false);
            })
    }

    pub fn local_rot(&self, scene: &Scene) -> Result<Quaternion<f32>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                scene.transform_data.get_unchecked(i).rot
            })
    }

    pub fn set_world_pos(&self, scene: &mut Scene, pos: Vector3<f32>) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                let local_pos = {
                    let data = scene.transform_data.get_unchecked(i);
                    let parent_data = data.parent_idx
                        .map(|idx| scene.transform_data.get_unchecked(idx));
                    scene.world_to_local_pos(parent_data, pos)
                };
                let data = scene.transform_data.get_unchecked_mut(i);
                data.pos = local_pos;
            })
    }

    pub fn world_pos(&self, scene: &Scene) -> Result<Vector3<f32>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                let data = scene.transform_data.get_unchecked(i);
                let parent_data = data.parent_idx
                    .map(|idx| scene.transform_data.get_unchecked(idx));
                scene.local_to_world_pos(parent_data, data.pos)
            })
    }

    pub fn set_world_rot(&self, scene: &mut Scene, rot: Quaternion<f32>) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                let local_rot = {
                    let data = scene.transform_data.get_unchecked(i);
                    let parent_data = data.parent_idx
                        .map(|idx| scene.transform_data.get_unchecked(idx));
                    scene.world_to_local_rot(parent_data, rot)
                };
                let data = scene.transform_data.get_unchecked_mut(i);
                data.rot = local_rot;
                data.ltw_valid.set(false);
            })
    }

    pub fn world_rot(&self, scene: &Scene) -> Result<Quaternion<f32>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                let data = scene.transform_data.get_unchecked(i);
                let parent_data = data.parent_idx
                    .map(|idx| scene.transform_data.get_unchecked(idx));
                scene.local_to_world_rot(parent_data, data.rot)
            })
    }

    pub fn is_valid(&self) -> bool {
        self.idx.get().is_some()
    }
}

const FRAME_TIME_MAX_SAMPLES: usize = 60;
const INITIAL_VERTEX_BUF_CAPACITY: usize = 1 * 1024 * 1024;
const INITIAL_INDEX_BUF_CAPACITY: usize = 1 * 1024 * 1024;

pub struct Scene {
    /// The OpenGL used for rendering, None if in headless mode.
    ctx: Option<GLContext>,
    /// Whether we are running on Intel graphics, don't try anything remotely
    /// fancy if so.
    buggy_intel: bool,
    window: Option<Window>,
    event_pump: Option<EventPump>,
    camera_data: Vec<CameraData>,
    mesh_data: Vec<MeshData>,
    material_data: Vec<MaterialData>,
    mrenderer_data: Vec<MeshRendererData>,
    shader_data: Vec<ShaderData>,
    behaviour_data: Vec<BehaviourData>,
    object_data: Vec<ObjectData>,
    transform_data: Vec<TransformData>,
    destroyed_cameras: Vec<usize>,
    destroyed_meshes: Vec<usize>,
    destroyed_materials: Vec<usize>,
    destroyed_mrenderers: Vec<usize>,
    destroyed_shaders: Vec<usize>,
    destroyed_behaviours: Vec<usize>,
    destroyed_objects: Vec<usize>,

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
    pub fn new(sdl: Sdl) -> Scene {
        let video_sys = sdl.video()
            .expect("Failed to initialize the Video subsystem");
        let event_pump = sdl.event_pump()
            .expect("Failed to obtain the SDL event pump");
        let window = video_sys.window("SDL2 Application", 800, 600)
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
            camera_data: Vec::new(),
            mesh_data: Vec::new(),
            material_data: Vec::new(),
            mrenderer_data: Vec::new(),
            shader_data: Vec::new(),
            behaviour_data: Vec::new(),
            object_data: Vec::new(),
            transform_data: Vec::new(),
            destroyed_cameras: Vec::new(),
            destroyed_meshes: Vec::new(),
            destroyed_materials: Vec::new(),
            destroyed_mrenderers: Vec::new(),
            destroyed_shaders: Vec::new(),
            destroyed_behaviours: Vec::new(),
            destroyed_objects: Vec::new(),

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
            camera_data: Vec::new(),
            mesh_data: Vec::new(),
            material_data: Vec::new(),
            mrenderer_data: Vec::new(),
            shader_data: Vec::new(),
            behaviour_data: Vec::new(),
            object_data: Vec::new(),
            transform_data: Vec::new(),
            destroyed_cameras: Vec::new(),
            destroyed_meshes: Vec::new(),
            destroyed_materials: Vec::new(),
            destroyed_mrenderers: Vec::new(),
            destroyed_shaders: Vec::new(),
            destroyed_behaviours: Vec::new(),
            destroyed_objects: Vec::new(),

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

    fn create_camera(&mut self, object: &Object) -> Result<Rc<Camera>> {
        let obj_idx = match object.idx.get() {
            Some(idx) => idx,
            None => {
                return Err(Error::ObjectDestroyed)
            }
        };
        let obj_data = &mut self.object_data[obj_idx];

        let rv = Rc::new(Camera { idx: Cell::new(None) });
        let data = CameraData {
            camera: rv.clone(),
            parent: obj_data.object.clone(),
            enabled: true,
            marked: false,
            order: 0,
            fovy: Deg(90.0),
            aspect: 1.0,
            near_clip: 1.0,
            far_clip: 1000.0,
            pos: Vector3::zero(),
            pv_matrix: Matrix4::one(),
            draw_cmds: Vec::new(),
        };
        self.camera_data.push(data);
        rv.idx.set(Some(self.camera_data.len() - 1));
        Ok(rv)
    }

    pub fn create_mesh(&mut self, vert_capacity: usize, indices_capacity: usize) -> Rc<Mesh> {
        if self.ctx.is_none() {
            // TODO: In the future implement some kind of dummy mesh?
            panic!("Tried to create mesh in headless mode.");
        }

        let rv = Rc::new(Mesh { idx: Cell::new(None) });
        let data = MeshData {
            object: rv.clone(),
            marked: false,
            vpos_vec: Vec::new(),
            vnorm_vec: Vec::new(),
            vcolour_vec: Vec::new(),
            indices_vec: Vec::new(),
            vertex_buf_dirty: false,
            vertex_buf_alloc: None,
            vertex_buf_capacity: vert_capacity,
            index_buf_dirty: false,
            index_buf_alloc: None,
            index_buf_capacity: indices_capacity,
            aabb_dirty: false,
            aabb_min: Vector3::zero(),
            aabb_max: Vector3::zero(),
        };
        self.mesh_data.push(data);
        rv.idx.set(Some(self.mesh_data.len() - 1));
        rv
    }

    pub fn create_material(&mut self, shader: Rc<Shader>) -> Result<Rc<Material>> {
        let rv = Rc::new(Material { idx: Cell::new(None) });
        let map = {
            let shader_data = try!(shader.idx.get()
                .ok_or(Error::ObjectDestroyed)
                .map(|i| unsafe { self.shader_data.get_unchecked(i) }));

            let mut map = HashMap::new();
            //for (name, _) in shader_data.program.uniforms() {
            //    // Names starting with "_" are reserved for our own use
            //    if name.starts_with("_") {
            //        map.insert(name.clone(), UnsafeCell::new(None));
            //    }
            //}
            map
        };

        let data = MaterialData {
            object: rv.clone(),
            marked: false,
            uniforms: map,
            shader: shader.clone(),
            colour: (1.0, 1.0, 1.0)
        };
        self.material_data.push(data);
        rv.idx.set(Some(self.material_data.len() - 1));
        Ok(rv)
    }

    fn create_mrenderer(&mut self, object: &Object) -> Result<Rc<MeshRenderer>> {
        let obj_idx = match object.idx.get() {
            Some(idx) => idx,
            None => {
                return Err(Error::ObjectDestroyed)
            }
        };
        let obj_data = &mut self.object_data[obj_idx];

        let rv = Rc::new(MeshRenderer { idx: Cell::new(None) });
        let data = MeshRendererData {
            object: rv.clone(),
            parent: obj_data.object.clone(),
            marked: false,
            enabled: true,
            mesh: None,
            material: None,
        };
        self.mrenderer_data.push(data);
        rv.idx.set(Some(self.mrenderer_data.len() - 1));
        Ok(rv)
    }

    pub fn create_shader(&mut self, vs_src: &str, fs_src: &str, _gs_src: Option<&str>) -> Rc<Shader> {
        if self.ctx.is_none() {
            // TODO: In the future implement some kind of dummy shader?
            panic!("Tried to create shader in headless mode.");
        }

        let rv = Rc::new(Shader { idx: Cell::new(None) });
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

        let data = ShaderData {
            object: rv.clone(),
            marked: false,
            program: program,
            obj_matrix_uniform: obj_matrix_uniform,
            cam_pos_uniform: cam_pos_uniform,
            cam_matrix_uniform: cam_matrix_uniform,
            colour_uniform: colour_uniform,
        };
        self.shader_data.push(data);
        rv.idx.set(Some(self.shader_data.len() - 1));
        rv
    }

    fn create_behaviour<T: BehaviourMessages + 'static>(&mut self, object: &Object) -> Result<Rc<RefCell<T>>> {
        let obj_idx = match object.idx.get() {
            Some(idx) => idx,
            None => {
                return Err(Error::ObjectDestroyed)
            }
        };
        let obj_data = &mut self.object_data[obj_idx];

        let t = T::create(Behaviour { idx: Cell::new(None) });
        let rv = Rc::new(RefCell::new(t));
        let data = BehaviourData {
            behaviour: rv.clone(),
            parent: obj_data.object.clone(),
            marked: false,
            is_new: true,
        };
        self.behaviour_data.push(data);
        rv.borrow().behaviour().idx.set(Some(self.behaviour_data.len() - 1));
        Ok(rv)
    }

    pub fn add_component<T: Component + 'static>(&mut self, object: &Object) -> Result<Rc<T>> {
        let obj_idx = match object.idx.get() {
            Some(idx) => idx,
            None => {
                return Err(Error::ObjectDestroyed)
            }
        };

        let id = TypeId::of::<T>();

        {
            let obj_data = &self.object_data[obj_idx];
            // Only overwrite if it doesn't exist or is marked
            let overwrite = obj_data.components.get(&id)
                .map_or(Ok(true), |comp| comp.marked(self));
            let overwrite = overwrite.expect("Destroyed component found still attached to object");
            if !overwrite {
                return Err(Error::Other);
            }
            if obj_data.marked {
                // TODO: dedicated error variant
                return Err(Error::Other)
            }
        }

        let comp = try!(T::init(self, object));

        let obj_data = &mut self.object_data[obj_idx];
        obj_data.components.insert(id, comp.clone());

        Ok(comp)
    }

    pub fn get_component<T: Component + 'static>(&self, object: &Object) -> Result<Rc<T>> {
        let obj_idx = match object.idx.get() {
            Some(idx) => idx,
            None => {
                return Err(Error::ObjectDestroyed)
            }
        };

        let id = TypeId::of::<T>();

        let comp = match self.object_data[obj_idx].components.get(&id) {
            Some(comp) => comp.clone(),
            None => {
                // TODO: dedicated error variant
                return Err(Error::Other)
            }
        };

        if comp.as_any().is::<T>() {
            unsafe {
                let raw: *mut AnyComponent = &*comp as *const _ as *mut _;
                std::mem::forget(comp);
                Ok(Rc::from_raw(raw as *mut T))
            }
        } else {
            // TODO: dedicated error variant
            Err(Error::Other)
        }
    }

    pub fn create_object(&mut self) -> Rc<Object> {
        let rv = Rc::new(Object { idx: Cell::new(None) });
        let obj_data = ObjectData {
            object: rv.clone(),
            components: HashMap::new(),
            marked: false,
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
        self.object_data.push(obj_data);
        self.transform_data.push(trans_data);
        rv.idx.set(Some(self.object_data.len() - 1));
        rv
    }

    pub fn destroy_camera(&mut self, camera: &Camera) {
        let camera_idx = match camera.idx.get() {
            Some(camera_idx) => camera_idx,
            None => {
                println!("[WARNING] destroy_camera called on a camera without a valid handle!");
                return
            }
        };
        let camera_data = unsafe {
            self.camera_data.get_unchecked_mut(camera_idx)
        };

        if !camera_data.marked {
            self.destroyed_cameras.push(camera_idx);
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

    pub fn destroy_material(&mut self, material: &Material) {
        let material_idx = match material.idx.get() {
            Some(material_idx) => material_idx,
            None => {
                println!("[WARNING] destroy_material called on a material without a valid handle!");
                return
            }
        };
        let material_data = unsafe {
            self.material_data.get_unchecked_mut(material_idx)
        };

        if !material_data.marked {
            self.destroyed_materials.push(material_idx);
            material_data.marked = true;
        }
    }

    pub fn destroy_mrenderer(&mut self, mrenderer: &MeshRenderer) {
        let mrenderer_idx = match mrenderer.idx.get() {
            Some(mrenderer_idx) => mrenderer_idx,
            None => {
                println!("[WARNING] destroy_mrenderer called on a mesh renderer without a valid handle!");
                return
            }
        };
        let mrenderer_data = &mut self.mrenderer_data[mrenderer_idx];

        if !mrenderer_data.marked {
            self.destroyed_mrenderers.push(mrenderer_idx);
            mrenderer_data.marked = true;
        }
    }

    pub fn destroy_shader(&mut self, shader: &Shader) {
        let shader_idx = match shader.idx.get() {
            Some(shader_idx) => shader_idx,
            None => {
                println!("[WARNING] destroy_shader called on a shader without a valid handle!");
                return
            }
        };
        let shader_data = unsafe {
            self.shader_data.get_unchecked_mut(shader_idx)
        };

        if !shader_data.marked {
            self.destroyed_shaders.push(shader_idx);
            shader_data.marked = true;
        }
    }

    pub fn destroy_behaviour(&mut self, behaviour: &Behaviour) {
        let bhav_idx = match behaviour.idx.get() {
            Some(bhav_idx) => bhav_idx,
            None => {
                println!("[WARNING] destroy_behaviour called on a behaviour without a valid handle!");
                return
            }
        };

        self.destroy_behaviour_internal(bhav_idx);
    }

    fn destroy_behaviour_internal(&mut self, idx: usize) {
        let bhav_data = unsafe {
            self.behaviour_data.get_unchecked_mut(idx)
        };

        if !bhav_data.marked {
            self.destroyed_behaviours.push(idx);
            bhav_data.marked = true;
        }
    }

    pub fn destroy_object(&mut self, object: &Object) {
        let object_idx = match object.idx.get() {
            Some(object_idx) => object_idx,
            None => {
                println!("[WARNING] destroy_object called on an object without a valid handle!");
                return
            }
        };

        self.destroy_object_internal(object_idx);
    }

    fn destroy_behaviour2<T: BehaviourMessages>(&mut self, behaviour: &T) {
        self.destroy_behaviour(behaviour.behaviour());
    }

    fn destroy_object_internal(&mut self, idx: usize) {
        let (was_marked, mut components) = {
            // FIXME: This is one example of many of workarounds to placate the
            // borrow checker. If I understand correctly, non-lexically based
            // lifetimes based on liveness should help in most cases. Update the
            // code when NLL is implemented in Rust.
            let obj_data = unsafe { self.object_data.get_unchecked_mut(idx) };
            // Swap map to placate the borrow checker
            let mut components = HashMap::new();
            std::mem::swap(&mut components, &mut obj_data.components);
            let was_marked = obj_data.marked;
            obj_data.marked = true;
            (was_marked, components)
        };

        for v in components.values() {
            v.destroy(self);
        }

        std::mem::swap(&mut components, &mut self.object_data[idx].components);

        // Swap vectors to placate the borrow checker
        let mut tmp_children = Vec::new();
        std::mem::swap(&mut tmp_children, &mut self.transform_data[idx].children);

        if !was_marked {
            self.destroyed_objects.push(idx);
            for &i in &tmp_children {
                Scene::destroy_object_internal(self, i);
            }
        }

        std::mem::swap(&mut tmp_children, &mut self.transform_data[idx].children);
    }

    fn debug_check(&self) {
        // For all objects, check the following:
        //   * The index is valid (i.e. `is_some()`)
        //   * The index corresponds to the correct data entry
        for i in 0..self.camera_data.len() {
            let data = unsafe { self.camera_data.get_unchecked(i) };
            let idx = data.camera.idx.get();
            assert!(idx.is_some(), "Invalid object handle found!");
            assert_eq!(idx.unwrap(), i);
        }
        for i in 0..self.object_data.len() {
            let data = unsafe { self.object_data.get_unchecked(i) };
            let idx = data.object.idx.get();
            assert!(idx.is_some(), "Invalid object handle found!");
            assert_eq!(idx.unwrap(), i);
        }
    }

    pub fn do_frame(&mut self) -> bool {
        if cfg!(debug_assertions) {
            self.debug_check();
        }

        let start_time = Instant::now();

        let mut idx = 0;
        while idx < self.behaviour_data.len() {
            let idx = post_add(&mut idx, 1);
            unsafe {
                let (is_new, cell) = {
                    let data = self.behaviour_data.get_unchecked(idx);
                    // Don't run `update()` on destroyed behaviours
                    if (*data).marked {
                        println!("Skipping behaviour {} because it's marked.", idx);
                        continue
                    }
                    ((*data).is_new, (&*(*data).behaviour) as *const AnyBehaviour)
                };
                let mut obj = (*cell).borrow_mut();
                if is_new {
                    obj.start(self);
                    let marked = {
                        let data = self.behaviour_data.get_unchecked_mut(idx);
                        (*data).is_new = false;
                        (*data).marked
                    };
                    // Check that the start function didn't immediately destroy the behaviour
                    if !marked {
                        obj.update(self);
                    }
                } else {
                    obj.update(self);
                }
            }
        }

        let mut i = 0;
        while i < self.destroyed_behaviours.len() {
            let i = post_add(&mut i, 1);
            unsafe {
                let idx = *self.destroyed_behaviours.get_unchecked(i);
                let cell = {
                    let data = self.behaviour_data.get_unchecked(idx);
                    (&*(*data).behaviour) as *const AnyBehaviour
                };
                (*cell).borrow_mut().destroy(self);
            }
        }

        let destroy_start_time = Instant::now();

        unsafe {
            let mut behaviour_data = Vec::new();
            let mut destroyed_behaviours = Vec::new();
            let mut camera_data = Vec::new();
            let mut destroyed_cameras = Vec::new();
            let mut mrenderer_data = Vec::new();
            let mut destroyed_mrenderers = Vec::new();

            std::mem::swap(&mut behaviour_data, &mut self.behaviour_data);
            std::mem::swap(&mut destroyed_behaviours, &mut self.destroyed_behaviours);
            std::mem::swap(&mut camera_data, &mut self.camera_data);
            std::mem::swap(&mut destroyed_cameras, &mut self.destroyed_cameras);
            std::mem::swap(&mut mrenderer_data, &mut self.mrenderer_data);
            std::mem::swap(&mut destroyed_mrenderers, &mut self.destroyed_mrenderers);

            Scene::cleanup_destroyed(
                &mut behaviour_data, &mut destroyed_behaviours,
                |x| x.marked,
                |x, idx| {
                    x.behaviour.borrow().behaviour().idx.set(idx);
                    if idx.is_none() {
                        // The object should not have been destroyed yet, so `unwrap()`
                        // is safe.
                        let obj_data = &mut self.object_data[x.parent.idx.get().unwrap()];
                        let id = x.behaviour.type_id();
                        let should_remove = obj_data.components.get(&id)
                            .map_or(false, |y| y.as_any() as *const _ == x.behaviour.as_any());
                        if should_remove {
                            obj_data.components.remove(&id);
                        }
                    }
                });
            Scene::cleanup_destroyed(
                &mut camera_data, &mut destroyed_cameras,
                |x| x.marked,
                |x, idx| {
                    x.camera.idx.set(idx);
                    if idx.is_none() {
                        // The object should not have been destroyed yet, so `unwrap()`
                        // is safe.
                        let obj_data = &mut self.object_data[x.parent.idx.get().unwrap()];
                        let id = x.camera.type_id();
                        let should_remove = obj_data.components.get(&id)
                            .map_or(false, |y| y.as_any() as *const _ == x.camera.as_any());
                        if should_remove {
                            obj_data.components.remove(&id);
                        }
                    }
                });
            Scene::cleanup_destroyed(
                &mut mrenderer_data, &mut destroyed_mrenderers,
                |x| x.marked,
                |x, idx| {
                    x.object.idx.set(idx);
                    if idx.is_none() {
                        // The object should not have been destroyed yet, so `unwrap()`
                        // is safe.
                        let obj_data = &mut self.object_data[x.parent.idx.get().unwrap()];
                        let id = x.object.type_id();
                        let should_remove = obj_data.components.get(&id)
                            .map_or(false, |y| y.as_any() as *const _ == x.object.as_any());
                        if should_remove {
                            obj_data.components.remove(&id);
                        }
                    }
                });

            self.cleanup_destroyed_objects();
            Scene::cleanup_destroyed(
                &mut self.mesh_data, &mut self.destroyed_meshes,
                |x| x.marked,
                |x, idx| x.object.idx.set(idx));
            Scene::cleanup_destroyed(
                &mut self.material_data, &mut self.destroyed_materials,
                |x| x.marked,
                |x, idx| x.object.idx.set(idx));
            Scene::cleanup_destroyed(
                &mut self.shader_data, &mut self.destroyed_shaders,
                |x| x.marked,
                |x, idx| x.object.idx.set(idx));

            std::mem::swap(&mut behaviour_data, &mut self.behaviour_data);
            std::mem::swap(&mut destroyed_behaviours, &mut self.destroyed_behaviours);
            std::mem::swap(&mut camera_data, &mut self.camera_data);
            std::mem::swap(&mut destroyed_cameras, &mut self.destroyed_cameras);
            std::mem::swap(&mut mrenderer_data, &mut self.mrenderer_data);
            std::mem::swap(&mut destroyed_mrenderers, &mut self.destroyed_mrenderers);
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

            let (min, max) = calc_aabb(&data.vpos_vec);
            data.aabb_dirty = false;
            data.aabb_min = min;
            data.aabb_max = max;
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

        for i in 0..self.transform_data.len() {
            let trans_data = &self.transform_data[i];

            // Check whether we need to recalculate the local-to-world matrix.
            let mut opt_data = Some(trans_data);
            let mut recalc = false;
            while let Some(data) = opt_data {
                if !trans_data.ltw_valid.get() {
                    recalc = true;
                    break
                }
                opt_data = data.parent_idx
                    .map(|idx| &self.transform_data[idx]);
            }

            if recalc {
                let (world_pos, world_rot) = {
                    let parent_data = trans_data.parent_idx
                        .map(|idx| &self.transform_data[idx]);
                    self.local_to_world_pos_rot(parent_data, trans_data.pos, trans_data.rot)
                };

                let local_to_world = (Matrix4::from_translation(world_pos)
                    * Matrix4::from(world_rot)).into();
                trans_data.ltw_matrix.set(local_to_world);
            }
        }

        // All local-to-world matrices have been recomputed where necessary,
        // mark as valid going in to the next frame.
        for trans_data in &self.transform_data {
            trans_data.ltw_valid.set(true);
        }

        // Calculate camera matrices and positions
        let mut camera_data = Vec::new();

        std::mem::swap(&mut camera_data, &mut self.camera_data);
        for camera in &mut camera_data {
            let trans_idx = camera.parent.idx.get()
                .expect("Camera parent object destroyed, the camera should have been destroyed as well at this point");
            let trans_data = &self.transform_data[trans_idx];
            let (cam_pos, cam_rot) = {
                let parent_data = trans_data.parent_idx
                    .map(|idx| &self.transform_data[idx]);
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
        std::mem::swap(&mut camera_data, &mut self.camera_data);

        for camera in &mut self.camera_data {
            camera.draw_cmds.clear();
        }

        // Generate draw commands
        for data in &self.mrenderer_data {
            if !data.enabled {
                continue;
            }

            let trans_idx = data.parent.idx.get();
            let trans_idx = match trans_idx {
                Some(trans_idx) => trans_idx,
                None => continue,
            };
            let trans_data = &self.transform_data[trans_idx];

            let mesh_idx = data.mesh
                .as_ref()
                .and_then(|x| x.idx.get());
            let mesh_idx = match mesh_idx {
                Some(mesh_idx) => mesh_idx,
                None => continue,
            };
            let mesh = &self.mesh_data[mesh_idx];

            let material_idx = data.material
                .as_ref()
                .and_then(|x| x.idx.get());
            let material_idx = match material_idx {
                Some(material_idx) => material_idx,
                None => continue,
            };
            let material = &self.material_data[material_idx];

            let shader_idx = material.shader.idx.get();
            let shader_idx = match shader_idx {
                Some(shader_idx) => shader_idx,
                None => continue
            };

            let vertex_buf_idx = mesh.vertex_buf_alloc
                .as_ref()
                .map(|&(ref idx, _)| *idx)
                .expect("Expected vertex buffer to be already allocated");
            let index_buf_idx = mesh.index_buf_alloc
                .as_ref()
                .map(|&(ref idx, _)| *idx)
                .expect("Expected index buffer to be already allocated");

            for camera in &mut self.camera_data {
                if !camera.enabled {
                    continue;
                }

                let points = aabb_points(mesh.aabb_min, mesh.aabb_max);

                let obj_matrix = Matrix4::from(trans_data.ltw_matrix.get());
                // Need to transpose since our matrix is actually PVM instead of MVP
                let mvp = (camera.pv_matrix * obj_matrix).transpose();

                if points_in_frustum(mvp, &points) {
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
        for camera in &mut self.camera_data {
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
        camera_render_order.extend(0..self.camera_data.len());

        // Sort cameras into the order we want to render them in
        camera_render_order.sort_by(|&a, &b| {
            let a_order = self.camera_data[a].order;
            let b_order = self.camera_data[b].order;
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
            let camera = &self.camera_data[camera_idx];
            let pv_matrix: [[f32; 4]; 4] = camera.pv_matrix.into();
            let cam_pos: [f32; 3] = camera.pos.into();

            gl::Clear(gl::DEPTH_BUFFER_BIT);
            for info in &camera.draw_cmds {
                if prev_shader_idx != Some(info.shader_idx) {
                    let shader = &self.shader_data[info.shader_idx];
                    gl::UseProgram(shader.program.program);
                    prev_shader_idx = Some(info.shader_idx);
                }
                if prev_material_idx != Some(info.material_idx) {
                    // TODO: support custom uniform values again
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

                let shader = &self.shader_data[info.shader_idx];
                let trans_data = &self.transform_data[info.trans_idx];
                let mesh = &self.mesh_data[info.mesh_idx];
                let material = &self.material_data[info.material_idx];

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

    //fn to_glium_uniform_value<'a>(&self, value: &UniformValue) -> glium::uniforms::UniformValue<'a> {
    //    match *value {
    //        UniformValue::Int(x) => glium::uniforms::UniformValue::SignedInt(x),
    //        UniformValue::UnsignedInt(x) => glium::uniforms::UniformValue::UnsignedInt(x),
    //        UniformValue::Float(x) => glium::uniforms::UniformValue::Float(x),
    //        UniformValue::Mat2(x) => glium::uniforms::UniformValue::Mat2(x),
    //        UniformValue::Mat3(x) => glium::uniforms::UniformValue::Mat3(x),
    //        UniformValue::Mat4(x) => glium::uniforms::UniformValue::Mat4(x),
    //        UniformValue::Vec2(x) => glium::uniforms::UniformValue::Vec2(x),
    //        UniformValue::Vec3(x) => glium::uniforms::UniformValue::Vec3(x),
    //        UniformValue::Vec4(x) => glium::uniforms::UniformValue::Vec4(x),
    //        UniformValue::IntVec2(x) => glium::uniforms::UniformValue::IntVec2(x),
    //        UniformValue::IntVec3(x) => glium::uniforms::UniformValue::IntVec3(x),
    //        UniformValue::IntVec4(x) => glium::uniforms::UniformValue::IntVec4(x),
    //        UniformValue::UIntVec2(x) => glium::uniforms::UniformValue::UnsignedIntVec2(x),
    //        UniformValue::UIntVec3(x) => glium::uniforms::UniformValue::UnsignedIntVec3(x),
    //        UniformValue::UIntVec4(x) => glium::uniforms::UniformValue::UnsignedIntVec4(x),
    //        UniformValue::Bool(x) => glium::uniforms::UniformValue::Bool(x),
    //        UniformValue::BoolVec2(x) => glium::uniforms::UniformValue::BoolVec2(x),
    //        UniformValue::BoolVec3(x) => glium::uniforms::UniformValue::BoolVec3(x),
    //        UniformValue::BoolVec4(x) => glium::uniforms::UniformValue::BoolVec4(x),
    //        UniformValue::Double(x) => glium::uniforms::UniformValue::Double(x),
    //        UniformValue::DoubleVec2(x) => glium::uniforms::UniformValue::DoubleVec2(x),
    //        UniformValue::DoubleVec3(x) => glium::uniforms::UniformValue::DoubleVec3(x),
    //        UniformValue::DoubleVec4(x) => glium::uniforms::UniformValue::DoubleVec4(x),
    //        UniformValue::DoubleMat2(x) => glium::uniforms::UniformValue::DoubleMat2(x),
    //        UniformValue::DoubleMat3(x) => glium::uniforms::UniformValue::DoubleMat3(x),
    //        UniformValue::DoubleMat4(x) => glium::uniforms::UniformValue::DoubleMat4(x),
    //        UniformValue::Int64(x) => glium::uniforms::UniformValue::Int64(x),
    //        UniformValue::Int64Vec2(x) => glium::uniforms::UniformValue::Int64Vec2(x),
    //        UniformValue::Int64Vec3(x) => glium::uniforms::UniformValue::Int64Vec3(x),
    //        UniformValue::Int64Vec4(x) => glium::uniforms::UniformValue::Int64Vec4(x),
    //        UniformValue::UInt64(x) => glium::uniforms::UniformValue::UnsignedInt64(x),
    //        UniformValue::UInt64Vec2(x) => glium::uniforms::UniformValue::UnsignedInt64Vec2(x),
    //        UniformValue::UInt64Vec3(x) => glium::uniforms::UniformValue::UnsignedInt64Vec3(x),
    //        UniformValue::UInt64Vec4(x) => glium::uniforms::UniformValue::UnsignedInt64Vec4(x)
    //    }
    //}

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
                .map(|idx| unsafe { self.transform_data.get_unchecked(idx) });
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
                .map(|idx| unsafe { self.transform_data.get_unchecked(idx) });
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

    pub fn set_object_parent(&mut self, object: &Object, parent: Option<&Object>) {
        let object_idx = object.idx.get();
        let parent_idx = parent.map(|o| o.idx.get());

        if parent_idx == Some(object_idx) {
            // Disallow parenting to self
            return
        }

        let object_idx = match object_idx {
            Some(idx) => idx,
            None => {
                // The object doesn't have a valid handle, stop.
                // TODO: maybe be less forgiving and just panic!()?
                return
            }
        };

        let parent_idx = match parent_idx {
            Some(None) => {
                // The parent doesn't have a valid handle, stop.
                // TODO: maybe be less forgiving and just panic!()?
                return
            },
            Some(opt_idx) => opt_idx,
            None => None,
        };

        if self.check_nop(object_idx, parent_idx) {
            // This parenting would be a no-op.
            return
        }

        if !self.check_parenting(object_idx, parent_idx) {
            // Performing this parenting is not allowed.
            // TODO: maybe be less forgiving and just panic!()?
            return
        }

        self.unparent(object_idx);
        self.reparent(object_idx, parent_idx);

        // Reparenting can change the local-to-world matrix in unpredictable
        // ways.
        let trans_data = &self.transform_data[object_idx];
        trans_data.ltw_valid.set(false);
    }

    fn check_nop(&self, object_idx: usize, parent_idx: Option<usize>) -> bool {
        let obj_trans_data = &self.transform_data[object_idx];
        obj_trans_data.parent_idx == parent_idx
    }

    fn check_parenting(&self, object_idx: usize, parent_idx: Option<usize>) -> bool {
        let parent_obj_data = parent_idx.map(|idx| &self.object_data[idx]);
        let parent_trans_data = parent_idx.map(|idx| &self.transform_data[idx]);
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
            opt_idx = self.transform_data[idx].parent_idx;
        }

        true
    }

    fn unparent(&mut self, object_idx: usize) {
        {
            let old_parent_idx = self.transform_data[object_idx].parent_idx;
            let old_parent_trans_data =
                old_parent_idx.map(|i| &mut self.transform_data[i]);

            if let Some(old_parent_trans_data) = old_parent_trans_data {
                old_parent_trans_data.children.iter()
                    .position(|&idx| idx == object_idx)
                    .map(|e| old_parent_trans_data.children.remove(e))
                    .expect("parent should contain child index");
            }
        }

        let trans_data = &mut self.transform_data[object_idx];
        trans_data.parent_idx = None;
    }

    fn reparent(&mut self, object_idx: usize, parent_idx: Option<usize>) {
        // Assume we currently have no parent, if parent_idx is `None`, we can
        // return early.
        let parent_idx = match parent_idx {
            Some(idx) => idx,
            None => return,
        };

        {
            let parent_trans_data = &mut self.transform_data[parent_idx];
            parent_trans_data.children.push(object_idx);
        }

        let trans_data = &mut self.transform_data[object_idx];
        trans_data.parent_idx = Some(parent_idx);
    }

    /// Fixes the object hierarchy while removing / moving an `ObjectData` entry
    ///
    ///   * `transform_data` - The `transform_data` field of the `Scene`
    ///   * `object_data` - The `object_data` field of the `Scene`
    ///   * `old_idx` - The old index of entry being removed / moved
    ///   * `new_idx` - The new index for the entry being moved, or `None` if
    ///      being removed
    unsafe fn fix_hierarchy(transform_data: &mut [TransformData],
                             object_data: &[ObjectData],
                             old_idx: usize,
                             new_idx: Option<usize>) {
        // TODO: use get_unchecked more? Or less?
        let obj_data = &object_data[old_idx];
        obj_data.object.idx.set(new_idx);
        // Update our parent's reference to us (if we have one)
        transform_data[old_idx].parent_idx.map(|idx| {
            let parent_data = transform_data.get_unchecked_mut(idx);
            let pos = {
                parent_data.children.iter()
                    .position(|&idx| idx == old_idx)
                    .expect("parent should contain child index")
            };
            match new_idx {
                Some(new_idx) => *parent_data.children.get_unchecked_mut(pos) = new_idx,
                None => { parent_data.children.remove(pos); }
            }
        });

        // Swap vectors to placate the borrow checker
        let mut tmp_children = Vec::new();
        std::mem::swap(&mut tmp_children, &mut transform_data[old_idx].children);

        // Update our children's reference to us
        for &idx in &tmp_children {
            let child_data = transform_data.get_unchecked_mut(idx);
            child_data.parent_idx = new_idx;
        }

        std::mem::swap(&mut tmp_children, &mut transform_data[old_idx].children);
    }

    unsafe fn cleanup_destroyed_objects(&mut self) {
        for &idx in &self.destroyed_objects {
            // Remove destroyed objects at the back of the list
            while self.object_data.last().map_or(false, |x| x.marked) {
                let old_idx = self.transform_data.len() - 1;
                Scene::fix_hierarchy(&mut self.transform_data, &self.object_data, old_idx, None);
                self.object_data.pop();
                self.transform_data.pop();
            }
            if idx >= self.transform_data.len() {
                continue
            }

            {
                let swapped_idx = self.transform_data.len() - 1;
                Scene::fix_hierarchy(&mut self.transform_data, &self.object_data, idx, None);
                Scene::fix_hierarchy(&mut self.transform_data, &self.object_data, swapped_idx, Some(idx));
            }
            self.object_data.swap_remove(idx);
            self.transform_data.swap_remove(idx);
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

    struct TestBehaviour {
        behaviour: Behaviour,
        id: u32
    }

    struct TestBehaviour2 {
        behaviour: Behaviour,
        id: u32,
    }

    impl BehaviourMessages for TestBehaviour {
        fn create(behaviour: Behaviour) -> TestBehaviour {
            TestBehaviour {
                behaviour: behaviour,
                id: 0
            }
        }

        fn start(&mut self, _scene: &mut Scene) {
        }

        fn update(&mut self, _scene: &mut Scene) {
        }

        fn destroy(&mut self, _scene: &mut Scene) {
        }

        fn behaviour(&self) -> &Behaviour {
            &self.behaviour
        }
    }

    impl BehaviourMessages for TestBehaviour2 {
        fn create(behaviour: Behaviour) -> TestBehaviour2 {
            TestBehaviour2 {
                behaviour: behaviour,
                id: 0
            }
        }

        fn start(&mut self, _scene: &mut Scene) {
        }

        fn update(&mut self, _scene: &mut Scene) {
        }

        fn destroy(&mut self, _scene: &mut Scene) {
        }

        fn behaviour(&self) -> &Behaviour {
            &self.behaviour
        }
    }

    /// Determine if two borrowed pointers point to the same thing.
    #[inline]
    fn ref_eq<T: ?Sized>(a: &T, b: &T) -> bool {
        (a as *const T) == (b as *const T)
    }

    /// Tests integrity of the Scene hierarchy after destroying objects.
    #[test]
    fn test_hierarchy_integrity() {
        let mut scene = Scene::new_headless();

        // Set up the hierarchy
        let root_obj = scene.create_object();

        let child1 = scene.create_object();
        scene.set_object_parent(&child1, Some(&root_obj));

        let child2 = scene.create_object();
        scene.set_object_parent(&child2, Some(&root_obj));

        let child11 = scene.create_object();
        scene.set_object_parent(&child11, Some(&child1));
        let child12 = scene.create_object();
        scene.set_object_parent(&child12, Some(&child1));
        let child13 = scene.create_object();
        scene.set_object_parent(&child13, Some(&child1));

        let child21 = scene.create_object();
        scene.set_object_parent(&child21, Some(&child2));
        let child22 = scene.create_object();
        scene.set_object_parent(&child22, Some(&child2));
        let child23 = scene.create_object();
        scene.set_object_parent(&child23, Some(&child2));

        scene.do_frame();

        // Verify it is what we expect
        assert_eq!(root_obj.get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child1)).ok(),
                   Some(true));
        assert_eq!(root_obj.get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child2)).ok(),
                   Some(true));

        assert_eq!(child1.get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child11)).ok(),
                   Some(true));
        assert_eq!(child1.get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child12)).ok(),
                   Some(true));
        assert_eq!(child1.get_child(&scene, 2)
                   .map(|x| ref_eq(&**x, &*child13)).ok(),
                   Some(true));

        assert_eq!(child2.get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child21)).ok(),
                   Some(true));
        assert_eq!(child2.get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child22)).ok(),
                   Some(true));
        assert_eq!(child2.get_child(&scene, 2)
                   .map(|x| ref_eq(&**x, &*child23)).ok(),
                   Some(true));

        // Destroy the objects and run a frame so the hierarchy is changed
        scene.destroy_object(&child2);
        scene.destroy_object(&child12);
        scene.do_frame();

        assert_eq!(root_obj.num_children(&scene).ok(), Some(1));
        assert_eq!(root_obj.get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child1)).ok(),
                   Some(true));
        assert_eq!(root_obj.get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child2)).ok(),
                   None);

        assert_eq!(child1.num_children(&scene).ok(), Some(2));
        assert_eq!(child1.get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child11)).ok(),
                   Some(true));
        assert_eq!(child1.get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child13)).ok(),
                   Some(true));

        assert_eq!(child2.num_children(&scene).ok(), None);
        assert_eq!(child2.get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child21)).ok(),
                   None);
        assert_eq!(child2.get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child22)).ok(),
                   None);
        assert_eq!(child2.get_child(&scene, 2)
                   .map(|x| ref_eq(&**x, &*child23)).ok(),
                   None);
    }

    /// Test to verify we cannot create cycles in the hierarchy.
    #[test]
    fn test_no_cycles() {
        let mut scene = Scene::new_headless();

        // Set up the hierarchy
        let root_obj = scene.create_object();

        let child_obj = scene.create_object();

        scene.set_object_parent(&child_obj, Some(&root_obj));
        // This should fail and do nothing
        scene.set_object_parent(&root_obj, Some(&child_obj));

        assert_eq!(root_obj.num_children(&scene).ok(), Some(1));
        assert_eq!(root_obj.get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child_obj)).ok(),
                   Some(true));

        assert_eq!(child_obj.num_children(&scene).ok(), Some(0));

        // Set up the hierarchy
        let obj1 = scene.create_object();
        let obj2 = scene.create_object();
        let obj3 = scene.create_object();
        let obj4 = scene.create_object();
        let obj5 = scene.create_object();
        scene.set_object_parent(&obj2, Some(&obj1));
        scene.set_object_parent(&obj3, Some(&obj2));
        scene.set_object_parent(&obj4, Some(&obj3));
        scene.set_object_parent(&obj5, Some(&obj4));
        // This should fail and do nothing
        scene.set_object_parent(&obj1, Some(&obj5));

        assert_eq!(obj5.num_children(&scene).ok(), Some(0));
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
        scene.set_object_parent(&obj2, Some(&obj1));
        scene.set_object_parent(&obj3, Some(&obj2));
        scene.set_object_parent(&obj4, Some(&obj3));
        scene.set_object_parent(&obj5, Some(&obj4));

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
        scene.set_object_parent(&obj2, Some(&obj1));

        obj1.set_local_rot(&mut scene, angles(-90.0, 0.0, 0.0)).unwrap();
        obj2.set_local_rot(&mut scene, angles(0.0, 0.0, -90.0)).unwrap();

        let tmp = obj2.world_rot(&scene).unwrap();
        let up = Vector3::new(0.0, 1.0, 0.0);
        let up_world = tmp * up;
        assert_relative_eq!(up_world.x, 1.0);
        assert_relative_eq!(up_world.y, 0.0);
        assert_relative_eq!(up_world.z, 0.0);
    }

    #[test]
    fn test_add_component() {
        let mut scene = Scene::new_headless();

        let obj1 = scene.create_object();

        // Test creating behaviour works
        let bhav = scene.add_component::<RefCell<TestBehaviour>>(&obj1).ok().unwrap();
        bhav.borrow_mut().id = 444;
        // Test we can retrieve the behaviour
        let bhav2 = scene.get_component::<RefCell<TestBehaviour>>(&obj1).ok().unwrap();
        // These behaviours should be identical
        assert_eq!(bhav.borrow().id, bhav2.borrow().id);

        // Test we cannot overwrite the old behaviour
        // TODO: check specific enum variant when added
        assert!(scene.add_component::<RefCell<TestBehaviour>>(&obj1).is_err());
        // Test we can add components of other types
        assert!(scene.add_component::<RefCell<TestBehaviour2>>(&obj1).is_ok());

        scene.destroy_behaviour(bhav.borrow().behaviour());

        // It should still be possible to retrieve the behaviour after it has
        // been marked for destruction
        scene.get_component::<RefCell<TestBehaviour>>(&obj1).ok().unwrap();

        // It should now be possible to overwrite the old behaviour
        let bhav3 = scene.add_component::<RefCell<TestBehaviour>>(&obj1).ok().unwrap();
        bhav3.borrow_mut().id = 555;

        // This behaviour should be different from the old one
        assert_ne!(bhav.borrow().id, bhav3.borrow().id);

        scene.destroy_behaviour(bhav3.borrow().behaviour());

        scene.do_frame();

        // It should no longer be possible to retrieve the behaviour in the next
        // frame
        // TODO: check specific enum variant when added
        assert!(scene.get_component::<RefCell<TestBehaviour>>(&obj1).is_err());

        // Other components should not be affected by one other component being
        // removed.
        let bhav4 = scene.get_component::<RefCell<TestBehaviour2>>(&obj1).unwrap();
        assert_eq!(bhav4.borrow().id, 0);

        scene.destroy_object(&obj1);

        // It should not be possible to add a new behaviour to a "marked"
        // object.
        // TODO: check specific enum variant when added
        assert!(scene.add_component::<RefCell<TestBehaviour>>(&obj1).is_err());

        let obj2 = scene.create_object();
        let bhav5 = scene.add_component::<RefCell<TestBehaviour>>(&obj2).ok().unwrap();

        scene.destroy_object(&obj2);

        scene.do_frame();

        // Destroying the object a component is attached to should destroy the
        // component.
        assert!(!bhav4.borrow().behaviour().is_valid());
        assert!(!bhav5.borrow().behaviour().is_valid());
    }
}
