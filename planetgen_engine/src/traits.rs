use {Object, Scene};
use error::Result;
use std::rc::Rc;

pub trait Component {
    fn init(scene: &mut Scene, object: &Object) -> Result<Rc<Self>> where Self: Sized ;

    fn marked(&self, scene: &Scene) -> Result<bool>;

    fn destroy(&self, scene: &mut Scene);
}
