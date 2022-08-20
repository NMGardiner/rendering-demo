use crate::*;

/// Destroy the underlying Vulkan objects associated with this object.
/// The object should no longer be used after `destroy` is called.
pub trait Destroy {
    fn destroy(&self, device: &Device);
}
