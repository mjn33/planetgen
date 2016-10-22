use glium::backend::glutin_backend::GlutinFacade;

pub struct Scene {
    /// The display used for rendering, None if in headless mode
    display: Option<GlutinFacade>
}

impl Scene {
    pub fn new(display: GlutinFacade) -> Scene {
        Scene {
            display: Some(display)
        }
    }

    /// Creates a new scene in "headless" mode (no graphical capabilities). Many
    /// operations will not work currently.
    pub fn new_headless() -> Scene {
        Scene {
            display: None
        }
    }
}
