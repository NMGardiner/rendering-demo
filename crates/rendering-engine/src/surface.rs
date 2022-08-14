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
    /// This function can error
    pub fn new(
        window: &dyn HasRawWindowHandle,
        instance: &Instance,
    ) -> Result<Self, Box<dyn Error>> {
        let surface_loader =
            ash::extensions::khr::Surface::new(instance.entry(), instance.handle());

        let surface = unsafe {
            ash_window::create_surface(instance.entry(), instance.handle(), window, None)?
        };

        log::debug!("Initialised surface.");

        Ok(Self {
            surface_loader: surface_loader,
            surface_handle: surface,
        })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader
                .destroy_surface(self.surface_handle, None);

            log::debug!("Destroyed surface.");
        }
    }
}
