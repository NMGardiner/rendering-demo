use std::error::Error;

use raw_window_handle::HasRawWindowHandle;

use crate::Instance;

/// A presentation surface, backed by a window.
pub struct Surface {
    surface_loader: ash::extensions::khr::Surface,
    surface_handle: ash::vk::SurfaceKHR,
}

impl Surface {
    /// Create a presentation surface for a given window.
    ///
    /// # Errors
    ///
    /// This function can error if `ash_window` fails to create the surface.
    pub fn new(
        window: &dyn HasRawWindowHandle,
        instance: &Instance,
    ) -> Result<Self, Box<dyn Error>> {
        let surface_loader = ash::extensions::khr::Surface::new(instance.entry(), instance);

        let surface =
            unsafe { ash_window::create_surface(instance.entry(), instance, window, None)? };

        Ok(Self {
            surface_loader,
            surface_handle: surface,
        })
    }

    pub fn handle(&self) -> &ash::vk::SurfaceKHR {
        &self.surface_handle
    }

    pub fn loader(&self) -> &ash::extensions::khr::Surface {
        &self.surface_loader
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader
                .destroy_surface(self.surface_handle, None);
        }
    }
}
