use std::{
    error::Error,
    ffi::CStr,
    ops::Deref,
    sync::{Arc, Mutex},
};

use gpu_allocator::vulkan::*;

use crate::*;

/// A logical device representing a single GPU within the system.
pub struct Device {
    device_handle: ash::Device,
    physical_device_handle: ash::vk::PhysicalDevice,
    graphics_queue: ash::vk::Queue,
    graphics_family_index: u32,

    memory_allocator: Option<Arc<Mutex<Allocator>>>,

    // Immediate submission resources.
    command_pool: ash::vk::CommandPool,
    command_buffer: ash::vk::CommandBuffer,
    completion_fence: ash::vk::Fence,
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

        let mut descriptor_indexing_features =
            ash::vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
                .descriptor_binding_partially_bound(true)
                .runtime_descriptor_array(true);

        let mut required_features = ash::vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut dynamic_rendering_features)
            .push_next(&mut synchronization2_features)
            .push_next(&mut descriptor_indexing_features)
            .features(core_features)
            .build();

        let physical_devices = unsafe { instance.enumerate_physical_devices()? };

        // Just select the first device that supports presentation, a graphics queue, and the required extensions.
        let mut graphics_family_index: Option<usize> = None;
        let physical_device = physical_devices.iter().find_map(|&device| {
            graphics_family_index = unsafe {
                instance
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

        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };

        // TODO: Should the application be fetching queues instead of doing it here?
        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };

        let device_properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let device_name = unsafe { CStr::from_ptr(device_properties.device_name.as_ptr()) };

        log::info!("Selected device: {}", device_name.to_str()?);

        let memory_allocator_info = AllocatorCreateDesc {
            instance: instance.deref().clone(),
            physical_device,
            device: device.clone(),
            debug_settings: Default::default(),
            buffer_device_address: false,
        };

        let memory_allocator = Arc::new(Mutex::new(Allocator::new(&memory_allocator_info)?));

        // Initialise immediate submission resources.
        let command_pool_info = ash::vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_family_index)
            .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };

        let command_buffer_info = ash::vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(ash::vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&command_buffer_info)?[0] };

        let completion_fence =
            unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None)? };

        Ok(Self {
            device_handle: device,
            physical_device_handle: physical_device,
            graphics_queue,
            graphics_family_index,
            memory_allocator: Some(memory_allocator),
            command_pool,
            command_buffer,
            completion_fence,
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

    pub fn memory_allocator(&self) -> Arc<Mutex<Allocator>> {
        self.memory_allocator.as_ref().unwrap().clone()
    }

    /// Immediately submit a function for execution on the device.
    ///
    /// # Errors
    ///
    /// This function can error if the given function returns an error, or if `ash` fails to start/end
    /// the command buffer, fails to submit to the graphics queue, fails to wait/reset the necessary fence,
    /// or fails to reset the immediate submission command pool.
    pub fn perform_immediate_submission<F>(&self, function: F) -> Result<(), Box<dyn Error>>
    where
        F: Fn(ash::vk::CommandBuffer) -> Result<(), Box<dyn Error>>,
    {
        let command_buffer_begin_info = ash::vk::CommandBufferBeginInfo::builder()
            .flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device_handle
                .begin_command_buffer(self.command_buffer, &command_buffer_begin_info)?;

            function(self.command_buffer)?;

            self.device_handle.end_command_buffer(self.command_buffer)?;

            let submission_command_buffers = [self.command_buffer];

            let submission_info =
                ash::vk::SubmitInfo::builder().command_buffers(&submission_command_buffers);

            self.device_handle.queue_submit(
                self.graphics_queue,
                &[*submission_info],
                self.completion_fence,
            )?;

            // Wait for the command buffer to be executed and reset the command pool for the next use.
            self.device_handle
                .wait_for_fences(&[self.completion_fence], true, std::u64::MAX)?;
            self.device_handle.reset_fences(&[self.completion_fence])?;

            self.device_handle
                .reset_command_pool(self.command_pool, ash::vk::CommandPoolResetFlags::empty())?;
        }

        Ok(())
    }

    pub fn perform_image_layout_transition(
        &self,
        command_buffer: &ash::vk::CommandBuffer,
        image: &ash::vk::Image,
        src_stage_mask: ash::vk::PipelineStageFlags2,
        src_access_mask: ash::vk::AccessFlags2,
        dst_stage_mask: ash::vk::PipelineStageFlags2,
        dst_access_mask: ash::vk::AccessFlags2,
        old_layout: ash::vk::ImageLayout,
        new_layout: ash::vk::ImageLayout,
        subresource_range: &ash::vk::ImageSubresourceRange,
    ) -> Result<(), Box<dyn Error>> {
        let barriers = [ash::vk::ImageMemoryBarrier2::builder()
            .src_stage_mask(src_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
            .image(*image)
            .subresource_range(*subresource_range)
            .build()];

        let dependency_info = ash::vk::DependencyInfo::builder()
            .dependency_flags(ash::vk::DependencyFlags::empty())
            .image_memory_barriers(&barriers);

        unsafe {
            self.device_handle
                .cmd_pipeline_barrier2(*command_buffer, &dependency_info);
        }

        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            // The memory allocator must be dropped manually.
            self.memory_allocator = None;

            // Destroy the immediate submission resources.
            self.device_handle
                .destroy_fence(self.completion_fence, None);
            self.device_handle
                .destroy_command_pool(self.command_pool, None);

            self.device_handle.destroy_device(None);
        }
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        self.handle()
    }
}
