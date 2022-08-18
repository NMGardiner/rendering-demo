use crate::*;

/// Destroy the underlying Vulkan objects associated with this object.
/// The object should no longer be used after `destroy` is called.
pub trait Destroy {
    fn destroy(&self, device: &Device);
}

/* TODO: Work on this, see if it's viable.
#[derive(Default)]
pub struct DestructionQueue {
    objects: Vec<Box<dyn Destroy>>,
}

impl DestructionQueue {
    pub fn push(&mut self, object: Box<dyn Destroy>) {
        self.objects.push(object);
    }

    pub fn destroy_all(&mut self, device: &Device) {
        for object in self.objects.iter() {
            object.destroy(device);
        }

        self.objects.clear();
    }
}

impl Drop for DestructionQueue {
    fn drop(&mut self) {
        if !self.objects.is_empty() {
            log::warn!("A destruction queue was dropped without destroying all objects!");
        }
    }
}
 */
