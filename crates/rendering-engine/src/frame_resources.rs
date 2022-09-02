use std::error::Error;

use crate::*;

/// An object storing the various resources (semaphores, command buffer etc) for
/// a single frame-in-flight.
pub struct FrameResources {
    command_pool: ash::vk::CommandPool,
    command_buffer: ash::vk::CommandBuffer,

    image_acquired_semaphore: ash::vk::Semaphore,
    render_finished_semaphore: ash::vk::Semaphore,
    render_finished_fence: ash::vk::Fence,

    deferred_deletion_queue: Vec<Box<dyn Destroy>>,
}

impl FrameResources {
    pub fn new(device: &Device) -> Result<Self, Box<dyn Error>> {
        let command_pool_info = ash::vk::CommandPoolCreateInfo::builder()
            .queue_family_index(device.graphics_family_index())
            .flags(ash::vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };

        let command_buffer_info = ash::vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(ash::vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&command_buffer_info)?[0] };

        let image_acquired_semaphore =
            unsafe { device.create_semaphore(&ash::vk::SemaphoreCreateInfo::default(), None)? };

        let render_finished_semaphore =
            unsafe { device.create_semaphore(&ash::vk::SemaphoreCreateInfo::default(), None)? };

        let render_finished_fence = unsafe {
            device.create_fence(
                &ash::vk::FenceCreateInfo::builder().flags(ash::vk::FenceCreateFlags::SIGNALED),
                None,
            )?
        };

        Ok(Self {
            command_pool,
            command_buffer,

            image_acquired_semaphore,
            render_finished_semaphore,
            render_finished_fence,

            deferred_deletion_queue: vec![],
        })
    }

    pub fn command_buffer(&self) -> &ash::vk::CommandBuffer {
        &self.command_buffer
    }

    pub fn begin_command_buffer(
        &self,
        device: &Device,
        usage_flags: ash::vk::CommandBufferUsageFlags,
    ) -> Result<(), Box<dyn Error>> {
        unsafe {
            Ok(device.begin_command_buffer(
                self.command_buffer,
                &ash::vk::CommandBufferBeginInfo::builder().flags(usage_flags),
            )?)
        }
    }

    pub fn end_command_buffer(&self, device: &Device) -> Result<(), Box<dyn Error>> {
        unsafe { Ok(device.end_command_buffer(self.command_buffer)?) }
    }

    pub fn image_acquired_semaphore(&self) -> &ash::vk::Semaphore {
        &self.image_acquired_semaphore
    }

    pub fn render_finished_semaphore(&self) -> &ash::vk::Semaphore {
        &self.render_finished_semaphore
    }

    pub fn render_finished_fence(&self) -> &ash::vk::Fence {
        &self.render_finished_fence
    }

    pub fn await_render_finished_fence(&self, device: &Device) -> Result<(), Box<dyn Error>> {
        unsafe { Ok(device.wait_for_fences(&[self.render_finished_fence], true, std::u64::MAX)?) }
    }

    pub fn reset_render_finished_fence(&self, device: &Device) -> Result<(), Box<dyn Error>> {
        unsafe { Ok(device.reset_fences(&[self.render_finished_fence])?) }
    }

    pub fn reset_command_buffer(&self, device: &Device) -> Result<(), Box<dyn Error>> {
        unsafe {
            Ok(device.reset_command_buffer(
                self.command_buffer,
                ash::vk::CommandBufferResetFlags::empty(),
            )?)
        }
    }

    /// Consume a destructible object, and add it to this frame's deferred deletion queue.
    /// Call [`FrameResources::process_deferred_deletion_queue`] when this frame is re-used to destroy
    /// the stored objects.
    pub fn defer_object_deletion<Type: Destroy + 'static>(&mut self, object: Type) {
        self.deferred_deletion_queue.push(Box::new(object));
    }

    /// Calls [`Destroy::destroy`] on each stored object, and clears the deletion queue.
    pub fn process_deferred_deletion_queue(&mut self, device: &Device) {
        for object in self.deferred_deletion_queue.iter_mut() {
            object.destroy(device);
        }

        self.deferred_deletion_queue.clear();
    }
}

impl Destroy for FrameResources {
    fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.image_acquired_semaphore, None);

            device.destroy_semaphore(self.render_finished_semaphore, None);

            device.destroy_fence(self.render_finished_fence, None);

            device.destroy_command_pool(self.command_pool, None);
        }
    }
}
