use cgmath;
use cgmath::{Deg, Euler, Matrix4, Quaternion, Rotation, Vector3};

use glium;
use glium::{IndexBuffer, Program, Surface, VertexBuffer};
use glium::backend::glutin_backend::GlutinFacade;

use num::{Zero, One};

use std;
use std::cell::{Cell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::error;
use std::fmt;
use std::rc::Rc;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// Indicates the operation failed since the underlying object was already
    /// destroyed.
    ObjectDestroyed,
    /// The specified child index was out of bounds
    BadChildIdx,
    /// Data given doesn't match size of buffer
    WrongBufferLength,
    /// Uniform name given not found in shader program
    BadUniformName,
    /// Other unspecified error.
    Other
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // For now just use error description, may change later with more enum
        // variants.
        write!(f, "{}", <Error as error::Error>::description(&self))
    }
}

impl error::Error for Error {
    fn description(&self) -> &str {
        match *self {
            Error::ObjectDestroyed => "Cannot perform operation on a destroyed object",
            Error::BadChildIdx => "Child index out of bounds",
            Error::WrongBufferLength => "Data given doesn't match size of buffer",
            Error::BadUniformName => "Uniform name given not found in shader program",
            Error::Other => "Unspecified error"
        }
    }
}

struct CameraData {
    /// Reference to the camera object.
    camera: Rc<Camera>,
    /// True when this camera should be used for rendering.
    enabled: bool,
    /// True when the camera has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    /// The direction the camera is facing.
    rot: Quaternion<f32>,
    /// The position of the camera in world coordinates.
    pos: Vector3<f32>,
    /// The vertical FOV for this camera.
    fovy: Deg<f32>,
    /// The aspect ratio for this camera.
    aspect: f32,
    /// Near clip plane distance.
    near_clip: f32,
    /// Far clip plane distance.
    far_clip: f32,
    // TODO: cache matrix transform for camera
    //cam_matrix: [[f32; 4]; 4],
}

/// A handle to a camera object for a scene.
pub struct Camera {
    idx: Cell<Option<usize>>
}

impl CameraData {
    fn calc_matrix(&self) -> [[f32; 4]; 4] {
        let cam_perspective = cgmath::perspective(self.fovy, self.aspect, self.near_clip, self.far_clip);
        let cam_matrix =
            cam_perspective *
            Matrix4::from(self.rot.invert()) *
            Matrix4::from_translation(-self.pos);
        cam_matrix.clone().into()
    }
}

impl Camera {
    pub fn is_valid(&self) -> bool {
        self.idx.get().is_some()
    }
}

#[derive(Copy, Clone)]
pub struct Vertex {
    pub vert_pos: [f32; 3],
}
implement_vertex!(Vertex, vert_pos);

struct MeshData {
    /// Reference to the mesh object.
    object: Rc<Mesh>,
    /// True when the mesh has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    vertex_buf: VertexBuffer<Vertex>,
    indices_buf: IndexBuffer<u16>,
}

pub struct Mesh {
    idx: Cell<Option<usize>>
}

impl Mesh {
    pub fn set_indices(&self, scene: &mut Scene, indices: &[u16]) -> Result<()> {
        let indices_buf = try!(self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .and_then(|i| unsafe {
                let indices_buf = &scene.mesh_data.get_unchecked(i).indices_buf;
                if indices.len() != indices_buf.len() {
                    Err(Error::WrongBufferLength)
                } else {
                    Ok(indices_buf)
                }
            }));

        indices_buf.write(indices);
        Ok(())
    }

    pub fn set_verts(&self, scene: &mut Scene, verts: &[Vertex]) -> Result<()> {
        let vertex_buf = try!(self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .and_then(|i| unsafe {
                let vertex_buf = &scene.mesh_data.get_unchecked(i).vertex_buf;
                if verts.len() != vertex_buf.len() {
                    Err(Error::WrongBufferLength)
                } else {
                    Ok(vertex_buf)
                }
            }));

        vertex_buf.write(verts);
        Ok(())
    }
}

struct ShaderData {
    /// Reference to the shader object.
    object: Rc<Shader>,
    /// True when the shader has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    program: Program,
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

pub trait Behaviour {
    fn create(object: Object) -> Self where Self: Sized;

    fn start(&mut self, scene: &mut Scene);

    fn update(&mut self, scene: &mut Scene);

    fn destroy(&mut self, scene: &mut Scene);

    fn object(&self) -> &Object;

    fn mesh(&self) -> Option<&Mesh>;

    fn material(&self) -> Option<&Material>;
}

struct ObjectData {
    /// Reference to the object behaviour.
    behaviour: Rc<RefCell<Behaviour>>,
    /// True when the object has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    /// True if the object has been newly created this current frame.
    is_new: bool
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

    pub fn get_child(&self, scene: &Scene, n: usize) -> Result<&Rc<RefCell<Behaviour>>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .and_then(|i| unsafe {
                scene.transform_data.get_unchecked(i).children.get(n)
                    .ok_or(Error::BadChildIdx)
            })
            .map(|&i| unsafe {
                &(*scene.object_data.get_unchecked(i).get()).behaviour
            })
    }

    pub fn set_local_pos(&self, scene: &mut Scene, pos: Vector3<f32>) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                scene.transform_data.get_unchecked_mut(i).pos = pos;
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
                scene.transform_data.get_unchecked_mut(i).rot = rot;
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

pub struct Scene {
    /// The display used for rendering, None if in headless mode.
    display: Option<GlutinFacade>,
    camera_data: Vec<CameraData>,
    mesh_data: Vec<MeshData>,
    material_data: Vec<MaterialData>,
    shader_data: Vec<ShaderData>,
    object_data: Vec<UnsafeCell<ObjectData>>,
    transform_data: Vec<TransformData>,
    destroyed_cameras: Vec<usize>,
    destroyed_meshes: Vec<usize>,
    destroyed_materials: Vec<usize>,
    destroyed_shaders: Vec<usize>,
    destroyed_objects: Vec<usize>,
    /// Temporary vector used in `local_to_world_pos_rot()`
    tmp_vec: UnsafeCell<Vec<(Vector3<f32>, Quaternion<f32>)>>
}

impl Scene {
    pub fn new(display: GlutinFacade) -> Scene {
        Scene {
            display: Some(display),
            camera_data: Vec::new(),
            mesh_data: Vec::new(),
            material_data: Vec::new(),
            shader_data: Vec::new(),
            object_data: Vec::new(),
            transform_data: Vec::new(),
            destroyed_cameras: Vec::new(),
            destroyed_meshes: Vec::new(),
            destroyed_materials: Vec::new(),
            destroyed_shaders: Vec::new(),
            destroyed_objects: Vec::new(),
            tmp_vec: UnsafeCell::new(Vec::new())
        }
    }

    /// Creates a new scene in "headless" mode (no graphical capabilities). Many
    /// operations will not work currently.
    pub fn new_headless() -> Scene {
        Scene {
            display: None,
            camera_data: Vec::new(),
            mesh_data: Vec::new(),
            material_data: Vec::new(),
            shader_data: Vec::new(),
            object_data: Vec::new(),
            transform_data: Vec::new(),
            destroyed_cameras: Vec::new(),
            destroyed_meshes: Vec::new(),
            destroyed_materials: Vec::new(),
            destroyed_shaders: Vec::new(),
            destroyed_objects: Vec::new(),
            tmp_vec: UnsafeCell::new(Vec::new())
        }
    }

    pub fn create_camera(&mut self) -> Rc<Camera> {
        let rv = Rc::new(Camera { idx: Cell::new(None) });
        let data = CameraData {
            camera: rv.clone(),
            enabled: true,
            marked: false,
            rot: Quaternion::from(Euler {
                x: Deg(0.0),
                y: Deg(0.0),
                z: Deg(0.0)
            }),
            pos: Vector3::new(0.0, 0.0, 0.0),
            fovy: Deg(90.0),
            aspect: 1.0,
            near_clip: 1.0,
            far_clip: 1000.0
        };
        self.camera_data.push(data);
        rv.idx.set(Some(self.camera_data.len() - 1));
        rv
    }

    pub fn create_mesh(&mut self, verts: &[Vertex], indices: &[u16]) -> Rc<Mesh> {
        let display = match self.display {
            Some(ref display) => display,
            None => {
                // TODO: In the future implement some kind of dummy mesh?
                panic!("Tried to create mesh in headless mode.");
            }
        };
        let rv = Rc::new(Mesh { idx: Cell::new(None) });
        let data = MeshData {
            object: rv.clone(),
            marked: false,
            vertex_buf: VertexBuffer::new(display, verts).unwrap(),
            indices_buf: IndexBuffer::new(display, glium::index::PrimitiveType::TrianglesList, indices).unwrap(),
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
            for (name, _) in shader_data.program.uniforms() {
                // Names starting with "_" are reserved for our own use
                if name.starts_with("_") {
                    map.insert(name.clone(), UnsafeCell::new(None));
                }
            }
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

    pub fn create_shader(&mut self, vs_src: &str, fs_src: &str, gs_src: Option<&str>) -> Rc<Shader> {
        let display = match self.display {
            Some(ref display) => display,
            None => {
                // TODO: In the future implement some kind of dummy shader?
                panic!("Tried to create shader in headless mode.");
            }
        };
        let rv = Rc::new(Shader { idx: Cell::new(None) });
        let program = Program::from_source(display, vs_src, fs_src, gs_src).unwrap();
        let data = ShaderData {
            object: rv.clone(),
            marked: false,
            program: program,
        };
        self.shader_data.push(data);
        rv.idx.set(Some(self.shader_data.len() - 1));
        rv
    }

    pub fn create_object<T: Behaviour + 'static>(&mut self) -> Rc<RefCell<T>> {
        let t: T = Behaviour::create(Object { idx: Cell::new(None) });
        let rv = Rc::new(RefCell::new(t));
        let obj_data = ObjectData {
            behaviour: rv.clone(),
            marked: false,
            is_new: true
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
        };
        self.object_data.push(UnsafeCell::new(obj_data));
        self.transform_data.push(trans_data);
        rv.borrow().object().idx.set(Some(self.object_data.len() - 1));
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

    pub fn destroy_object(&mut self, object: &Object) {
        let object_idx = match object.idx.get() {
            Some(object_idx) => object_idx,
            None => {
                println!("[WARNING] destroy_object called on an object without a valid handle!");
                return
            }
        };

        //self.destroy_object_internal(object_idx);
        Scene::destroy_object_internal(&self.object_data,
                                       &self.transform_data,
                                       &mut self.destroyed_objects,
                                       object_idx);
    }

    fn destroy_object_internal(object_data: &Vec<UnsafeCell<ObjectData>>,
                               transform_data: &Vec<TransformData>,
                               destroyed_objects: &mut Vec<usize>,
                               idx: usize) {
        let obj_data = unsafe { &mut *object_data.get_unchecked(idx).get() };
        let trans_data = unsafe { transform_data.get_unchecked(idx) };

        if !obj_data.marked {
            obj_data.marked = true;
            destroyed_objects.push(idx);
            for &i in &trans_data.children {
                Scene::destroy_object_internal(object_data, transform_data, destroyed_objects, i);
            }
        }
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
            let data = unsafe { &*self.object_data.get_unchecked(i).get() };
            let idx = data.behaviour.borrow().object().idx.get();
            assert!(idx.is_some(), "Invalid object handle found!");
            assert_eq!(idx.unwrap(), i);
        }
    }

    pub fn do_frame(&mut self) -> bool {
        fn post_add<T: Copy + std::ops::Add<Output=T>>(a: &mut T, b: T) -> T {
            let c = *a;
            *a = *a + b;
            c
        }

        if cfg!(debug_assertions) {
            self.debug_check();
        }

        let mut idx = 0;
        while idx < self.object_data.len() {
            let idx = post_add(&mut idx, 1);
            unsafe {
                let (is_new, cell) = {
                    let data = self.object_data.get_unchecked(idx).get();
                    // Don't run `update()` on destroyed objects
                    if (*data).marked {
                        println!("Skipping object {} because it's marked.", idx);
                        continue
                    }
                    ((*data).is_new, (&*(*data).behaviour) as *const RefCell<Behaviour>)
                };
                let mut obj = (*cell).borrow_mut();
                if is_new {
                    obj.start(self);
                    let data = self.object_data.get_unchecked(idx).get();
                    (*data).is_new = false;
                    // Check that the start function didn't immediately destroy the object
                    if !(*data).marked {
                        obj.update(self);
                    }
                } else {
                    obj.update(self);
                }
            }
        }

        let mut i = 0;
        while i < self.destroyed_objects.len() {
            let i = post_add(&mut i, 1);
            unsafe {
                let idx = *self.destroyed_objects.get_unchecked(i);
                let cell = {
                    let data = self.object_data.get_unchecked(idx);
                    (&*(*data.get()).behaviour) as *const RefCell<Behaviour>
                };
                (*cell).borrow_mut().destroy(self);
            }
        }

        // TODO: don't call unwrap on this since it breaks headless mode
        let mut target = self.display.as_ref().unwrap().draw();
        target.clear_color_and_depth((0.8, 0.8, 0.8, 1.0), 1.0);

        //for data in &self.object_data {
        let mut idx = 0;
        while idx < self.object_data.len() {
            let idx = post_add(&mut idx, 1);
            let obj_data = unsafe { self.object_data.get_unchecked(idx) };
            let trans_data = unsafe { self.transform_data.get_unchecked(idx) };
            let behaviour = unsafe { (*obj_data.get()).behaviour.borrow() };
            let mesh = behaviour.mesh()
                .and_then(|x| x.idx.get())
                .and_then(|idx| unsafe { Some(self.mesh_data.get_unchecked(idx)) });
            let mesh = match mesh {
                Some(mesh) => mesh,
                None => continue
            };

            let material = behaviour.material()
                .and_then(|x| x.idx.get())
                .and_then(|idx| unsafe { Some(self.material_data.get_unchecked(idx)) });
            let material = match material {
                Some(material) => material,
                None => continue
            };

            let shader = material.shader.idx.get()
                .and_then(|idx| unsafe { Some(self.shader_data.get_unchecked(idx)) });
            let shader = match shader {
                Some(shader) => shader,
                None => continue
            };

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

            let (world_pos, world_rot) = unsafe {
                let parent_data = trans_data.parent_idx
                    .map(|idx| self.transform_data.get_unchecked(idx));
                self.local_to_world_pos_rot(parent_data, trans_data.pos, trans_data.rot)
            };

            // TODO: multiple cameras
            let cam_matrix = self.camera_data[0].calc_matrix();

            let local_to_world = (Matrix4::from_translation(world_pos)
                * Matrix4::from(world_rot)).into();

            let colour = [material.colour.0, material.colour.1, material.colour.2];

            struct TmpUniforms<'a, 'b> {
                scene: &'a Scene,
                uniforms: &'b HashMap<String, UnsafeCell<Option<UniformValue>>>,
                cam_matrix: [[f32; 4]; 4],
                obj_matrix: [[f32; 4]; 4],
                colour: [f32; 3]
            }

            impl<'a, 'b> glium::uniforms::Uniforms for TmpUniforms<'a, 'b> {
                fn visit_values<'c, F: FnMut(&str, glium::uniforms::UniformValue<'c>)>(&'c self, mut f: F) {
                    f("_obj_matrix", glium::uniforms::UniformValue::Mat4(self.obj_matrix));
                    f("_cam_matrix", glium::uniforms::UniformValue::Mat4(self.cam_matrix));
                    f("_colour", glium::uniforms::UniformValue::Vec3(self.colour));
                    for (name, entry) in self.uniforms {
                        let entry = unsafe { (&*entry.get()) };
                        entry.as_ref()
                            .map(|v| self.scene.to_glium_uniform_value(&v))
                            .map(|v| f(&name, v));
                    }
                }
            }

            let uniforms = TmpUniforms {
                scene: self,
                uniforms: &material.uniforms,
                cam_matrix: cam_matrix,
                obj_matrix: local_to_world,
                colour: colour
            };

            target.draw(&mesh.vertex_buf, &mesh.indices_buf, &shader.program, &uniforms, &draw_params).unwrap();
        }

        target.finish().unwrap();

        for ev in self.display.as_ref().unwrap().poll_events() {
            match ev {
                glium::glutin::Event::Closed => return false,
                _ => ()
            }
        }

        unsafe {
            self.cleanup_destroyed_objects();
            Scene::cleanup_destroyed(
                &mut self.camera_data, &mut self.destroyed_cameras,
                |x| x.marked,
                |x, idx| x.camera.idx.set(idx));
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
        }

        true
    }

    fn to_glium_uniform_value<'a>(&self, value: &UniformValue) -> glium::uniforms::UniformValue<'a> {
        match *value {
            UniformValue::Int(x) => glium::uniforms::UniformValue::SignedInt(x),
            UniformValue::UnsignedInt(x) => glium::uniforms::UniformValue::UnsignedInt(x),
            UniformValue::Float(x) => glium::uniforms::UniformValue::Float(x),
            UniformValue::Mat2(x) => glium::uniforms::UniformValue::Mat2(x),
            UniformValue::Mat3(x) => glium::uniforms::UniformValue::Mat3(x),
            UniformValue::Mat4(x) => glium::uniforms::UniformValue::Mat4(x),
            UniformValue::Vec2(x) => glium::uniforms::UniformValue::Vec2(x),
            UniformValue::Vec3(x) => glium::uniforms::UniformValue::Vec3(x),
            UniformValue::Vec4(x) => glium::uniforms::UniformValue::Vec4(x),
            UniformValue::IntVec2(x) => glium::uniforms::UniformValue::IntVec2(x),
            UniformValue::IntVec3(x) => glium::uniforms::UniformValue::IntVec3(x),
            UniformValue::IntVec4(x) => glium::uniforms::UniformValue::IntVec4(x),
            UniformValue::UIntVec2(x) => glium::uniforms::UniformValue::UnsignedIntVec2(x),
            UniformValue::UIntVec3(x) => glium::uniforms::UniformValue::UnsignedIntVec3(x),
            UniformValue::UIntVec4(x) => glium::uniforms::UniformValue::UnsignedIntVec4(x),
            UniformValue::Bool(x) => glium::uniforms::UniformValue::Bool(x),
            UniformValue::BoolVec2(x) => glium::uniforms::UniformValue::BoolVec2(x),
            UniformValue::BoolVec3(x) => glium::uniforms::UniformValue::BoolVec3(x),
            UniformValue::BoolVec4(x) => glium::uniforms::UniformValue::BoolVec4(x),
            UniformValue::Double(x) => glium::uniforms::UniformValue::Double(x),
            UniformValue::DoubleVec2(x) => glium::uniforms::UniformValue::DoubleVec2(x),
            UniformValue::DoubleVec3(x) => glium::uniforms::UniformValue::DoubleVec3(x),
            UniformValue::DoubleVec4(x) => glium::uniforms::UniformValue::DoubleVec4(x),
            UniformValue::DoubleMat2(x) => glium::uniforms::UniformValue::DoubleMat2(x),
            UniformValue::DoubleMat3(x) => glium::uniforms::UniformValue::DoubleMat3(x),
            UniformValue::DoubleMat4(x) => glium::uniforms::UniformValue::DoubleMat4(x),
            UniformValue::Int64(x) => glium::uniforms::UniformValue::Int64(x),
            UniformValue::Int64Vec2(x) => glium::uniforms::UniformValue::Int64Vec2(x),
            UniformValue::Int64Vec3(x) => glium::uniforms::UniformValue::Int64Vec3(x),
            UniformValue::Int64Vec4(x) => glium::uniforms::UniformValue::Int64Vec4(x),
            UniformValue::UInt64(x) => glium::uniforms::UniformValue::UnsignedInt64(x),
            UniformValue::UInt64Vec2(x) => glium::uniforms::UniformValue::UnsignedInt64Vec2(x),
            UniformValue::UInt64Vec3(x) => glium::uniforms::UniformValue::UnsignedInt64Vec3(x),
            UniformValue::UInt64Vec4(x) => glium::uniforms::UniformValue::UnsignedInt64Vec4(x)
        }
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
            world_rot = world_rot * data.rot;
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
    }

    fn check_nop(&self, object_idx: usize, parent_idx: Option<usize>) -> bool {
        let obj_trans_data = &self.transform_data[object_idx];
        obj_trans_data.parent_idx == parent_idx
    }

    fn check_parenting(&self, object_idx: usize, parent_idx: Option<usize>) -> bool {
        let parent_obj_data = parent_idx.map(|idx| unsafe { &*self.object_data[idx].get() });
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
                             object_data: &[UnsafeCell<ObjectData>],
                             old_idx: usize,
                             new_idx: Option<usize>) {
        // TODO: use get_unchecked more? Or less?
        let obj_data = &*object_data[old_idx].get();
        obj_data.behaviour.borrow().object().idx.set(new_idx);
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
            while self.object_data.last().map_or(false, |x| (*x.get()).marked) {
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
                                         set_idx: G)
        where F: Fn(&T) -> bool, G: Fn(&T, Option<usize>) {
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

    struct TestObject {
        object: Object,
        id: u32
    }

    impl Behaviour for TestObject {
        fn create(object: Object) -> TestObject {
            TestObject {
                object: object,
                id: 0
            }
        }

        fn start(&mut self, _scene: &mut Scene) {
        }

        fn update(&mut self, _scene: &mut Scene) {
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
        let root_obj = scene.create_object::<TestObject>();
        root_obj.borrow_mut().id = 1;

        let child1 = scene.create_object::<TestObject>();
        child1.borrow_mut().id = 2;
        scene.set_object_parent(child1.borrow().object(), Some(root_obj.borrow().object()));

        let child2 = scene.create_object::<TestObject>();
        child2.borrow_mut().id = 3;
        scene.set_object_parent(child2.borrow().object(), Some(root_obj.borrow().object()));

        let child11 = scene.create_object::<TestObject>();
        child11.borrow_mut().id = 4;
        scene.set_object_parent(child11.borrow().object(), Some(child1.borrow().object()));
        let child12 = scene.create_object::<TestObject>();
        child12.borrow_mut().id = 5;
        scene.set_object_parent(child12.borrow().object(), Some(child1.borrow().object()));
        let child13 = scene.create_object::<TestObject>();
        child13.borrow_mut().id = 6;
        scene.set_object_parent(child13.borrow().object(), Some(child1.borrow().object()));

        let child21 = scene.create_object::<TestObject>();
        child21.borrow_mut().id = 7;
        scene.set_object_parent(child21.borrow().object(), Some(child2.borrow().object()));
        let child22 = scene.create_object::<TestObject>();
        child22.borrow_mut().id = 8;
        scene.set_object_parent(child22.borrow().object(), Some(child2.borrow().object()));
        let child23 = scene.create_object::<TestObject>();
        child23.borrow_mut().id = 9;
        scene.set_object_parent(child23.borrow().object(), Some(child2.borrow().object()));

        scene.do_frame();

        // Verify it is what we expect
        assert_eq!(root_obj.borrow().object().get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child1)).ok(),
                   Some(true));
        assert_eq!(root_obj.borrow().object().get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child2)).ok(),
                   Some(true));

        assert_eq!(child1.borrow().object().get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child11)).ok(),
                   Some(true));
        assert_eq!(child1.borrow().object().get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child12)).ok(),
                   Some(true));
        assert_eq!(child1.borrow().object().get_child(&scene, 2)
                   .map(|x| ref_eq(&**x, &*child13)).ok(),
                   Some(true));

        assert_eq!(child2.borrow().object().get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child21)).ok(),
                   Some(true));
        assert_eq!(child2.borrow().object().get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child22)).ok(),
                   Some(true));
        assert_eq!(child2.borrow().object().get_child(&scene, 2)
                   .map(|x| ref_eq(&**x, &*child23)).ok(),
                   Some(true));

        // Destroy the objects and run a frame so the hierarchy is changed
        scene.destroy_object(child2.borrow().object());
        scene.destroy_object(child12.borrow().object());
        scene.do_frame();

        assert_eq!(root_obj.borrow().object().num_children(&scene).ok(), Some(1));
        assert_eq!(root_obj.borrow().object().get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child1)).ok(),
                   Some(true));
        assert_eq!(root_obj.borrow().object().get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child2)).ok(),
                   None);

        assert_eq!(child1.borrow().object().num_children(&scene).ok(), Some(2));
        assert_eq!(child1.borrow().object().get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child11)).ok(),
                   Some(true));
        assert_eq!(child1.borrow().object().get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child13)).ok(),
                   Some(true));

        assert_eq!(child2.borrow().object().num_children(&scene).ok(), None);
        assert_eq!(child2.borrow().object().get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child21)).ok(),
                   None);
        assert_eq!(child2.borrow().object().get_child(&scene, 1)
                   .map(|x| ref_eq(&**x, &*child22)).ok(),
                   None);
        assert_eq!(child2.borrow().object().get_child(&scene, 2)
                   .map(|x| ref_eq(&**x, &*child23)).ok(),
                   None);
    }

    /// Test to verify we cannot create cycles in the hierarchy.
    #[test]
    fn test_no_cycles() {
        let mut scene = Scene::new_headless();

        // Set up the hierarchy
        let root_obj = scene.create_object::<TestObject>();
        root_obj.borrow_mut().id = 1;

        let child_obj = scene.create_object::<TestObject>();
        child_obj.borrow_mut().id = 2;

        scene.set_object_parent(child_obj.borrow().object(), Some(root_obj.borrow().object()));
        // This should fail and do nothing
        scene.set_object_parent(root_obj.borrow().object(), Some(child_obj.borrow().object()));

        assert_eq!(root_obj.borrow().object().num_children(&scene).ok(), Some(1));
        assert_eq!(root_obj.borrow().object().get_child(&scene, 0)
                   .map(|x| ref_eq(&**x, &*child_obj)).ok(),
                   Some(true));

        assert_eq!(child_obj.borrow().object().num_children(&scene).ok(), Some(0));

        // Set up the hierarchy
        let obj1 = scene.create_object::<TestObject>();
        let obj2 = scene.create_object::<TestObject>();
        let obj3 = scene.create_object::<TestObject>();
        let obj4 = scene.create_object::<TestObject>();
        let obj5 = scene.create_object::<TestObject>();
        scene.set_object_parent(obj2.borrow().object(), Some(obj1.borrow().object()));
        scene.set_object_parent(obj3.borrow().object(), Some(obj2.borrow().object()));
        scene.set_object_parent(obj4.borrow().object(), Some(obj3.borrow().object()));
        scene.set_object_parent(obj5.borrow().object(), Some(obj4.borrow().object()));
        // This should fail and do nothing
        scene.set_object_parent(obj1.borrow().object(), Some(obj5.borrow().object()));

        assert_eq!(obj5.borrow().object().num_children(&scene).ok(), Some(0));
    }

    /// Tests objects are transformed correctly
    #[test]
    fn test_obj_transforms() {
        let mut scene = Scene::new_headless();

        let obj1 = scene.create_object::<TestObject>();
        let obj2 = scene.create_object::<TestObject>();
        let obj3 = scene.create_object::<TestObject>();
        let obj4 = scene.create_object::<TestObject>();
        let obj5 = scene.create_object::<TestObject>();
        scene.set_object_parent(obj2.borrow().object(), Some(obj1.borrow().object()));
        scene.set_object_parent(obj3.borrow().object(), Some(obj2.borrow().object()));
        scene.set_object_parent(obj4.borrow().object(), Some(obj3.borrow().object()));
        scene.set_object_parent(obj5.borrow().object(), Some(obj4.borrow().object()));

        fn angles(x: f32, y: f32, z: f32) -> Quaternion<f32> {
            Quaternion::from(Euler { x: Deg(x), y: Deg(y), z: Deg(z) })
        }
        obj1.borrow().object().set_local_pos(&mut scene, Vector3::new(10.0, 0.0, 0.0)).unwrap();
        obj1.borrow().object().set_local_rot(&mut scene, angles(0.0, 0.0, 45.0)).unwrap();

        obj2.borrow().object().set_local_pos(&mut scene, Vector3::new(10.0, 0.0, 0.0)).unwrap();
        obj2.borrow().object().set_local_rot(&mut scene, angles(0.0, 0.0, 45.0)).unwrap();

        obj3.borrow().object().set_local_pos(&mut scene, Vector3::new(10.0, 0.0, 0.0)).unwrap();
        obj3.borrow().object().set_local_rot(&mut scene, angles(0.0, 0.0, 45.0)).unwrap();

        obj4.borrow().object().set_local_pos(&mut scene, Vector3::new(10.0, 0.0, 0.0)).unwrap();
        obj4.borrow().object().set_local_rot(&mut scene, angles(0.0, 0.0, 45.0)).unwrap();

        obj5.borrow().object().set_local_pos(&mut scene, Vector3::new(10.0, 0.0, 0.0)).unwrap();
        obj5.borrow().object().set_local_rot(&mut scene, angles(0.0, 0.0, 45.0)).unwrap();

        obj4.borrow().object().set_world_pos(&mut scene, Vector3::zero()).unwrap();
        obj4.borrow().object().set_world_rot(&mut scene, Quaternion::one()).unwrap();

        let tmp = obj1.borrow().object().world_pos(&scene).unwrap();
        assert_relative_eq!(tmp.x, 10.0);
        assert_relative_eq!(tmp.y, 0.0);
        assert_relative_eq!(tmp.z, 0.0);

        let tmp = obj2.borrow().object().world_pos(&scene).unwrap();
        assert_relative_eq!(tmp.x, 17.071067812);
        assert_relative_eq!(tmp.y, 7.071067812);
        assert_relative_eq!(tmp.z, 0.0);

        let tmp = obj3.borrow().object().world_pos(&scene).unwrap();
        assert_relative_eq!(tmp.x, 17.071067812);
        assert_relative_eq!(tmp.y, 17.071067812);
        assert_relative_eq!(tmp.z, 0.0);

        let tmp = obj4.borrow().object().world_pos(&scene).unwrap();
        assert_relative_eq!(tmp.x, 0.0);
        assert_relative_eq!(tmp.y, 0.0);
        assert_relative_eq!(tmp.z, 0.0);

        let tmp = obj5.borrow().object().world_pos(&scene).unwrap();
        assert_relative_eq!(tmp.x, 10.0, max_relative = 1.05);
        assert_relative_eq!(tmp.y, 0.0, max_relative = 1.05);
        assert_relative_eq!(tmp.z, 0.0);
    }
}
