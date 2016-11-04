use cgmath::{Deg, Euler, Quaternion, Rotation, Vector3};

use glium::{IndexBuffer, Program, VertexBuffer};
use glium::backend::glutin_backend::GlutinFacade;

use num::{Zero, One};

use std::cell::{Cell, RefCell, UnsafeCell};
use std::error;
use std::fmt;
use std::rc::Rc;

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// Indicates the operation failed since the underlying object was already
    /// destroyed.
    ObjectDestroyed,
    /// The specified child index was out of bounds
    BadChildIdx,
    /// Data given doesn't match size of buffer
    WrongBufferLength,
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

impl Camera {
    pub fn is_valid(&self) -> bool {
        self.idx.get().is_some()
    }
}

#[derive(Copy, Clone)]
pub struct Vertex {
    vert_pos: [f32; 3],
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

pub trait Behaviour {
    fn create(object: Object) -> Self where Self: Sized;

    fn start(&mut self, scene: &mut Scene);

    fn update(&mut self, scene: &mut Scene);

    fn destroy(&mut self, scene: &mut Scene);

    fn object(&self) -> &Object;
}

struct ObjectData {
    /// Local rotation.
    rot: Quaternion<f32>,
    /// Local position.
    pos: Vector3<f32>,
    /// Reference to the object behaviour.
    behaviour: Rc<RefCell<Behaviour>>,
    /// List of indices of the children of this object.
    children: Vec<usize>,
    /// Index of this object's parent.
    parent_idx: Option<usize>,
    /// True when the object has been marked for destruction at the end of the
    /// frame.
    marked: bool,
    /// True if the object has been newly created this current frame.
    is_new: bool
}

pub struct Object {
    idx: Cell<Option<usize>>
}

impl Object {
    pub fn num_children(&self, scene: &Scene) -> Result<usize> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe { (*scene.object_data.get_unchecked(i).get()).children.len() })
    }

    pub fn get_child(&self, scene: &Scene, n: usize) -> Result<&Rc<RefCell<Behaviour>>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .and_then(|i| unsafe {
                (*scene.object_data.get_unchecked(i).get()).children.get(n)
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
                (*scene.object_data.get_unchecked(i).get()).pos = pos;
            })
    }

    pub fn local_pos(&self, scene: &Scene) -> Result<Vector3<f32>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                (*scene.object_data.get_unchecked(i).get()).pos
            })
    }

    pub fn set_local_rot(&self, scene: &mut Scene, rot: Quaternion<f32>) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                (*scene.object_data.get_unchecked(i).get()).rot = rot;
            })
    }

    pub fn local_rot(&self, scene: &Scene) -> Result<Quaternion<f32>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                (*scene.object_data.get_unchecked(i).get()).rot
            })
    }

    pub fn set_world_pos(&self, scene: &mut Scene, pos: Vector3<f32>) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                let data = &mut *scene.object_data.get_unchecked(i).get();
                let parent_data = data.parent_idx
                    .map(|idx| &*scene.object_data.get_unchecked(idx).get());
                let local_pos = scene.world_to_local_pos(parent_data, pos);
                data.pos = local_pos;
            })
    }

    pub fn world_pos(&self, scene: &Scene) -> Result<Vector3<f32>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                let data = &*scene.object_data.get_unchecked(i).get();
                let parent_data = data.parent_idx
                    .map(|idx| &*scene.object_data.get_unchecked(idx).get());
                scene.local_to_world_pos(parent_data, data.pos)
            })
    }

    pub fn set_world_rot(&self, scene: &mut Scene, rot: Quaternion<f32>) -> Result<()> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                let data = &mut *scene.object_data.get_unchecked(i).get();
                let parent_data = data.parent_idx
                    .map(|idx| &*scene.object_data.get_unchecked(idx).get());
                let local_rot = scene.world_to_local_rot(parent_data, rot);
                data.rot = local_rot;
            })
    }

    pub fn world_rot(&self, scene: &Scene) -> Result<Quaternion<f32>> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe {
                let data = &*scene.object_data.get_unchecked(i).get();
                let parent_data = data.parent_idx
                    .map(|idx| &*scene.object_data.get_unchecked(idx).get());
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
    destroyed_cameras: Vec<usize>,
    mesh_data: Vec<MeshData>,
    destroyed_meshes: Vec<usize>,
    shader_data: Vec<ShaderData>,
    destroyed_shaders: Vec<usize>,
    object_data: Vec<UnsafeCell<ObjectData>>,
    destroyed_objects: Vec<usize>,
    /// Temporary vector used in `local_to_world_pos_rot()`
    tmp_vec: UnsafeCell<Vec<(Vector3<f32>, Quaternion<f32>)>>
}

impl Scene {
    pub fn new(display: GlutinFacade) -> Scene {
        Scene {
            display: Some(display),
            camera_data: Vec::new(),
            destroyed_cameras: Vec::new(),
            mesh_data: Vec::new(),
            destroyed_meshes: Vec::new(),
            shader_data: Vec::new(),
            destroyed_shaders: Vec::new(),
            object_data: Vec::new(),
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
            destroyed_cameras: Vec::new(),
            mesh_data: Vec::new(),
            destroyed_meshes: Vec::new(),
            shader_data: Vec::new(),
            destroyed_shaders: Vec::new(),
            object_data: Vec::new(),
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
            indices_buf: IndexBuffer::new(display, ::glium::index::PrimitiveType::TrianglesList, indices).unwrap(),
        };
        self.mesh_data.push(data);
        rv.idx.set(Some(self.mesh_data.len() - 1));
        rv
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
        let data = ObjectData {
            rot: Quaternion::from(Euler {
                x: Deg(0.0),
                y: Deg(0.0),
                z: Deg(0.0)
            }),
            pos: Vector3::new(0.0, 0.0, 0.0),
            behaviour: rv.clone(),
            children: Vec::new(),
            parent_idx: None,
            marked: false,
            is_new: true
        };
        self.object_data.push(UnsafeCell::new(data));
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

        self.destroy_object_internal(object_idx);
    }

    fn destroy_object_internal(&mut self, idx: usize) {
        let object_data = unsafe {
            let data = self.object_data.get_unchecked(idx);
            &mut (*data.get())
        };

        if !object_data.marked {
            object_data.marked = true;
            self.destroyed_objects.push(idx);
            for &i in &object_data.children {
                self.destroy_object_internal(i);
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

    fn do_frame(&mut self) {
        fn post_add<T: Copy + ::std::ops::Add<Output=T>>(a: &mut T, b: T) -> T {
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
                &mut self.shader_data, &mut self.destroyed_shaders,
                |x| x.marked,
                |x, idx| x.object.idx.set(idx));
        }
    }

    fn local_to_world_pos_rot(&self,
                              parent_data: Option<&ObjectData>,
                              input_pos: Vector3<f32>,
                              input_rot: Quaternion<f32>)
                              -> (Vector3<f32>, Quaternion<f32>) {
        let mut tmp_vec = unsafe { &mut *self.tmp_vec.get() };
        tmp_vec.clear();
        let mut opt_data = parent_data;
        while let Some(data) = opt_data {
            tmp_vec.push((data.pos, data.rot));
            opt_data = data.parent_idx
                .map(|idx| unsafe { &*self.object_data.get_unchecked(idx).get() });
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

    fn local_to_world_pos(&self, parent_data: Option<&ObjectData>, input_pos: Vector3<f32>)
                          -> Vector3<f32> {
        let (world_pos, _) = self.local_to_world_pos_rot(parent_data, input_pos, Quaternion::one());
        world_pos
    }

    fn local_to_world_rot(&self, parent_data: Option<&ObjectData>, input_rot: Quaternion<f32>)
                          -> Quaternion<f32> {
        let mut opt_data = parent_data;
        let mut world_rot = input_rot;
        while let Some(data) = opt_data {
            world_rot = world_rot * data.rot;
            opt_data = data.parent_idx
                .map(|idx| unsafe { &*self.object_data.get_unchecked(idx).get() });
        }
        world_rot
    }

    fn world_to_local_pos_rot(&self,
                              parent_data: Option<&ObjectData>,
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

    fn world_to_local_pos(&self, parent_data: Option<&ObjectData>, input_pos: Vector3<f32>)
                          -> Vector3<f32> {
        let (local_pos, _) = self.world_to_local_pos_rot(parent_data, input_pos, Quaternion::one());
        local_pos
    }

    fn world_to_local_rot(&self, parent_data: Option<&ObjectData>, input_rot: Quaternion<f32>)
                          -> Quaternion<f32> {
        let world_rot = self.local_to_world_rot(parent_data, Quaternion::one());
        let inv_world_rot = world_rot.invert();
        let local_rot = inv_world_rot * input_rot;
        local_rot
    }

    pub fn set_object_parent(&mut self, object: &Object, parent: Option<&Object>) {
        if parent.map(|o| o.idx.get()) == Some(object.idx.get()) {
            // Disallow parenting to self
            return
        }
        if parent.map_or(false, |o| !o.is_valid()) || !object.is_valid() {
            // Either one of these objects hasn't got a valid handle, stop.
            // TODO: maybe be less forgiving and just panic!()?
            return
        }
        let child_idx = object.idx.get().unwrap();
        let child_data = unsafe {
            //&mut (*self.object_data.get())[child_idx as usize]
            &mut (*self.object_data[child_idx].get())
        };
        let (parent_idx, parent_data) = match parent {
            Some(parent) => {
                let parent_idx = parent.idx.get().unwrap();
                let parent_data = unsafe {
                    //&mut (*self.object_data.get())[parent_idx as usize]
                    &mut (*self.object_data[parent_idx].get())
                };
                (Some(parent_idx), Some(parent_data))
            },
            None => {
                //// Unparent
                //child_data.parent_idx = None;
                //return
                (None, None)
            }
        };
        if parent_data.as_ref().map_or(false, |x| x.marked) {
            // Can't parent to something marked for destruction
            // TODO: maybe be less forgiving and just panic!()?
            return
        }

        if child_data.parent_idx == parent_idx {
            // No change
            return
        }

        let mut tmp_idx = parent_data.as_ref()
            .and_then(|parent_data| parent_data.parent_idx);
        loop {
            tmp_idx = match tmp_idx {
                Some(tmp_idx) if tmp_idx == child_idx => {
                    // Performing this parenting would create a loop, bail.
                    // TODO: maybe be less forgiving and just panic!()?
                    return
                },
                Some(tmp_idx) => {
                    let tmp_data = unsafe { &*self.object_data[tmp_idx].get() };
                    tmp_data.parent_idx
                },
                None => break,
            }
        }

        let old_parent_data =
            child_data.parent_idx.as_ref().map(|&i| unsafe {
                //&mut (*self.object_data.get())[i as usize]
                &mut (*self.object_data[i].get())
            });

        if let Some(old_parent_data) = old_parent_data {
            old_parent_data.children.iter()
                .position(|&idx| idx == child_idx)
                .map(|e| old_parent_data.children.remove(e))
                .expect("parent should contain child index");
        }
        if let Some(parent_data) = parent_data {
            parent_data.children.push(child_idx);
        }
        child_data.parent_idx = parent_idx;
    }


    /// Fixes the object hierarchy while removing / moving an `ObjectData` entry
    ///   * `object_data` - The `object_data` field of the `Scene`
    ///   * `data` - The object data entry which has been removed / moved
    ///   * `old_idx` - The old index of entry being removed / moved
    ///   * `new_idx` - The new index for the entry being moved, or `None` if
    ///      being removed
    unsafe fn fix_hierarchy(object_data: &[UnsafeCell<ObjectData>],
                            data: &UnsafeCell<ObjectData>,
                            old_idx: usize,
                            new_idx: Option<usize>) {
        let data = &*data.get();
        data.behaviour.borrow().object().idx.set(new_idx);
        // Update our parent's reference to us (if we have one)
        data.parent_idx.map(|idx| {
            let parent_data = &mut *object_data.get_unchecked(idx).get();
            let pos = parent_data.children.iter()
                .position(|&idx| idx == old_idx)
                .expect("parent should contain child index");
            match new_idx {
                Some(new_idx) => *parent_data.children.get_unchecked_mut(pos) = new_idx,
                None => { parent_data.children.remove(pos); }
            }
        });
        // Update our children's reference to us
        for &idx in &data.children {
            let child_data = &mut *object_data.get_unchecked(idx).get();
            child_data.parent_idx = new_idx;
        }
    }

    unsafe fn cleanup_destroyed_objects(&mut self) {
        for &idx in &self.destroyed_objects {
            // Remove destroyed objects at the back of the list
            while self.object_data.last().map_or(false, |x| (*x.get()).marked) {
                let removed = self.object_data.pop().unwrap();
                let old_idx = self.object_data.len();
                Scene::fix_hierarchy(&self.object_data, &removed, old_idx, None);
            }
            if idx >= self.object_data.len() {
                continue
            }

            {
                let removed = self.object_data.get_unchecked(idx);
                let swapped_idx = self.object_data.len() - 1;
                let swapped = self.object_data.get_unchecked(swapped_idx);
                Scene::fix_hierarchy(&self.object_data, removed, idx, None);
                Scene::fix_hierarchy(&self.object_data, swapped, swapped_idx, Some(idx));
            }
            self.object_data.swap_remove(idx);
            self.object_data.get_unchecked(idx);
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
