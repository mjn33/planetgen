use cgmath::{Deg, Euler, Quaternion, Vector3};

use glium::backend::glutin_backend::GlutinFacade;

use std::cell::Cell;
use std::rc::Rc;

struct CameraData {
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

pub struct Scene {
    /// The display used for rendering, None if in headless mode.
    display: Option<GlutinFacade>,
    camera_data: Vec<CameraData>,
    destroyed_cameras: Vec<usize>
}

impl Scene {
    pub fn new(display: GlutinFacade) -> Scene {
        Scene {
            display: Some(display),
            camera_data: Vec::new(),
            destroyed_cameras: Vec::new()
        }
    }

    /// Creates a new scene in "headless" mode (no graphical capabilities). Many
    /// operations will not work currently.
    pub fn new_headless() -> Scene {
        Scene {
            display: None,
            camera_data: Vec::new(),
            destroyed_cameras: Vec::new()
        }
    }

    pub fn create_camera(&mut self) -> Rc<Camera> {
        let rv = Rc::new(Camera { idx: Cell::new(None) });
        let data = CameraData {
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
}
