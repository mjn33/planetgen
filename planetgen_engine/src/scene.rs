use cgmath::{Deg, Euler, Quaternion, Vector3};

use glium::backend::glutin_backend::GlutinFacade;

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
    //mesh: Option<Mesh>
}

pub struct Object {
    idx: Cell<Option<usize>>
}

impl Object {
    pub fn num_children(&self, scene: &Scene) -> Result<usize> {
        self.idx.get()
            .ok_or(Error::ObjectDestroyed)
            .map(|i| unsafe { (*scene.object_data[i].get()).children.len() })
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
    object_data: Vec<UnsafeCell<ObjectData>>,
    destroyed_objects: Vec<usize>
}

impl Scene {
    pub fn new(display: GlutinFacade) -> Scene {
        Scene {
            display: Some(display),
            camera_data: Vec::new(),
            destroyed_cameras: Vec::new(),
            object_data: Vec::new(),
            destroyed_objects: Vec::new()
        }
    }

    /// Creates a new scene in "headless" mode (no graphical capabilities). Many
    /// operations will not work currently.
    pub fn new_headless() -> Scene {
        Scene {
            display: None,
            camera_data: Vec::new(),
            destroyed_cameras: Vec::new(),
            object_data: Vec::new(),
            destroyed_objects: Vec::new()
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
            Scene::cleanup_destroyed(
                &mut self.object_data, &mut self.destroyed_objects,
                |x| (*x.get()).marked,
                |x, idx| (&*x.get()).behaviour.borrow().object().idx.set(idx));
            Scene::cleanup_destroyed(
                &mut self.camera_data, &mut self.destroyed_cameras,
                |x| x.marked,
                |x, idx| x.camera.idx.set(idx));
        }
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
            set_idx(&swapped, Some(idx));
        }
        destroyed_items.clear();
    }
}
