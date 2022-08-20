use std::{error::Error, ffi::CStr};

use crate::*;

/// A logical device representing a single GPU within the system.
pub struct Device {
    device_handle: ash::Device,
    physical_device_handle: ash::vk::PhysicalDevice,
    graphics_queue: ash::vk::Queue,
    graphics_family_index: u32,
}

impl Device {
    /// Creates a new logical device if a suitable physical device is found.
    /// Also checks for presentation support if a surface is given.
    ///
    /// # Errors
    ///
    /// This function will error if `ash` fails to get the available physical devices, if `ash` fails
    /// to create the logical device, or if there are no suitable physical devices present.
    pub fn new(instance: &Instance, surface: Option<&Surface>) -> Result<Self, Box<dyn Error>> {
        let mut required_extensions = vec![
            ash::extensions::khr::DynamicRendering::name(),
            ash::extensions::khr::Synchronization2::name(),
        ];

        // Only add the swapchain extension if we're rendering to a surface.
        if surface.is_some() {
            required_extensions.push(ash::extensions::khr::Swapchain::name());
        }

        // The DeviceCreateInfo needs the extension names as a *const c_char slice.
        let enabled_extensions = required_extensions
            .iter()
            .map(|extension| extension.as_ptr())
            .collect::<Vec<_>>();

        let core_features = ash::vk::PhysicalDeviceFeatures::default();

        let mut dynamic_rendering_features =
            ash::vk::PhysicalDeviceDynamicRenderingFeaturesKHR::builder().dynamic_rendering(true);

        let mut synchronization2_features =
            ash::vk::PhysicalDeviceSynchronization2Features::builder().synchronization2(true);

        let mut required_features = ash::vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut synchronization2_features)
            .features(core_features)
            .build();

        let physical_devices = unsafe { instance.handle().enumerate_physical_devices()? };

        // Just select the first device that supports presentation, a graphics queue, and the required extensions.
        let mut graphics_family_index: Option<usize> = None;
        let physical_device = physical_devices.iter().find_map(|&device| {
            graphics_family_index = unsafe {
                instance
                    .handle()
                    .get_physical_device_queue_family_properties(device)
                    .iter()
                    .enumerate()
                    .position(|(index, queue_family)| {
                        let supports_graphics = queue_family
                            .queue_flags
                            .contains(ash::vk::QueueFlags::GRAPHICS);

                        let mut device_is_suitable = supports_graphics;

                        // Check if presentation to the surface is supported, if one is given.
                        if let Some(surface) = surface {
                            device_is_suitable &= surface
                                .loader()
                                .get_physical_device_surface_support(
                                    device,
                                    index.try_into().unwrap(),
                                    *surface.handle(),
                                )
                                .unwrap_or(false);
                        }

                        device_is_suitable
                    })
            };

            let supports_required_extensions = unsafe {
                let supported_extensions = instance
                    .handle()
                    .enumerate_device_extension_properties(device)
                    .unwrap_or_default();

                let supported_extension_names = supported_extensions
                    .iter()
                    .map(|extension| CStr::from_ptr(extension.extension_name.as_ptr()))
                    .collect::<Vec<_>>();

                required_extensions.iter().all(|required_extension| {
                    supported_extension_names.contains(required_extension)
                })
            };

            // TODO: Check that the required features are supported.
            if graphics_family_index.is_some() && supports_required_extensions {
                Some(device)
            } else {
                None
            }
        });

        if physical_device.is_none() {
            return Err(
                String::from("No suitable devices found. Initialisation can not continue").into(),
            );
        }

        let physical_device = physical_device.unwrap();
        let graphics_family_index = graphics_family_index.unwrap() as u32;

        // Only a single (graphics) queue is used for now, but there needs to be one of these per queue.
        let queue_infos = [ash::vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(graphics_family_index)
            .queue_priorities(&[1.0])
            .build()];

        let device_info = ash::vk::DeviceCreateInfo::builder()
            .push_next(&mut required_features)
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_extensions);

        let device = unsafe {
            instance
                .handle()
                .create_device(physical_device, &device_info, None)?
        };

        // TODO: Should the application be fetching queues instead of doing it here?
        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };

        let device_properties = unsafe {
            instance
                .handle()
                .get_physical_device_properties(physical_device)
        };

        let device_name = unsafe { CStr::from_ptr(device_properties.device_name.as_ptr()) };

        log::info!("Selected device: {}", device_name.to_str()?);

        Ok(Self {
            device_handle: device,
            physical_device_handle: physical_device,
            graphics_queue,
            graphics_family_index,
        })
    }

    pub fn handle(&self) -> &ash::Device {
        &self.device_handle
    }

    pub fn physical_device_handle(&self) -> &ash::vk::PhysicalDevice {
        &self.physical_device_handle
    }

    pub fn graphics_queue(&self) -> &ash::vk::Queue {
        &self.graphics_queue
    }

    pub fn graphics_family_index(&self) -> u32 {
        self.graphics_family_index
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device_handle.destroy_device(None);
        }
    }
}
