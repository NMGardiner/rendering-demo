use std::error::Error;

use crate::*;

/// An object storing the various resources (semaphores, command buffers etc) for
/// a given frame-in-flight.
pub struct FrameResources {
    command_pool: ash::vk::CommandPool,
    command_buffer: ash::vk::CommandBuffer,

    image_acquired_semaphore: ash::vk::Semaphore,
    render_finished_semaphore: ash::vk::Semaphore,
    render_finished_fence: ash::vk::Fence,
}

impl FrameResources {
    pub fn new(device: &Device) -> Result<Self, Box<dyn Error>> {
        let command_pool_info = ash::vk::CommandPoolCreateInfo::builder()
            .queue_family_index(device.graphics_family_index())
            .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            device
                .handle()
                .create_command_pool(&command_pool_info, None)?
        };

        let command_buffer_info = ash::vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(ash::vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            device
                .handle()
                .allocate_command_buffers(&command_buffer_info)?[0]
        };

        let image_acquired_semaphore = unsafe {
            device
                .handle()
                .create_semaphore(&ash::vk::SemaphoreCreateInfo::default(), None)?
        };

        let render_finished_semaphore = unsafe {
            device
                .handle()
                .create_semaphore(&ash::vk::SemaphoreCreateInfo::default(), None)?
        };

        let render_finished_fence = unsafe {
            device.handle().create_fence(
                &ash::vk::FenceCreateInfo::builder().flags(ash::vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };

        log::debug!("Initialise frame resources.");

        Ok(Self {
            command_pool: command_pool,
            command_buffer: command_buffer,

            image_acquired_semaphore: image_acquired_semaphore,
            render_finished_semaphore: render_finished_semaphore,
            render_finished_fence: render_finished_fence,
        })
    }

    pub fn begin_command_buffer(
        &self,
        device: &Device,
        usage_flags: ash::vk::CommandBufferUsageFlags,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            Ok(device.handle().begin_command_buffer(
                self.command_buffer,
                &ash::vk::CommandBufferBeginInfo::builder().flags(usage_flags),
            )?)
        }
    }

    pub fn end_command_buffer(&self, device: &Device) -> Result<(), Box<dyn Error>> {
        unsafe { Ok(device.handle().end_command_buffer(self.command_buffer)?) }
    }

    pub fn await_render_finished_fence(&self, device: &Device) -> Result<(), Box<dyn Error>> {
        unsafe {
            Ok(device.handle().wait_for_fences(
                &[self.render_finished_fence],
                true,
                std::u64::MAX,
            )?)
        }
    }

    pub fn reset_render_finished_fence(&self, device: &Device) -> Result<(), Box<dyn Error>> {
        unsafe {
            Ok(device
                .handle()
                .reset_fences(&[self.render_finished_fence])?)
        }
    }

    pub fn reset_command_buffer(&self, device: &Device) -> Result<(), Box<dyn Error>> {
        unsafe {
            Ok(device.handle().reset_command_buffer(
                self.command_buffer,
                ash::vk::CommandBufferResetFlags::empty(),
            )?)
        }
    }
}

impl Destroy for FrameResources {
    fn destroy(&self, device: &Device) {
        unsafe {
            device
                .handle()
                .destroy_semaphore(self.image_acquired_semaphore, None);

            device
                .handle()
                .destroy_semaphore(self.render_finished_semaphore, None);

            device
                .handle()
                .destroy_fence(self.render_finished_fence, None);

            device
                .handle()
                .destroy_command_pool(self.command_pool, None);
        }

        log::debug!("Destroyed frame resources.");
    }
}
