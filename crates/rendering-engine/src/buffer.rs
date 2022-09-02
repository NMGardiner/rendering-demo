use std::error::Error;

use gpu_allocator::{vulkan::*, MemoryLocation};

use crate::*;

/// A Vulkan buffer object, allocated using gpu_allocator.
pub struct Buffer {
    buffer_handle: ash::vk::Buffer,
    allocation: Allocation,
}

impl Buffer {
    /// Create a new buffer of the given size, with the given usage flags and memory location.
    /// The allocated memory will be mapped and bound.
    ///
    /// # Errors
    ///
    /// This function can error if `ash` fails to create the buffer, bind the buffer memory, or if `gpu_allocator`
    /// fails to allocate the requested memory.
    pub fn new(
        device: &Device,
        size: u64,
        usage: ash::vk::BufferUsageFlags,
        location: MemoryLocation,
    ) -> Result<Self, Box<dyn Error>> {
        let buffer_info = ash::vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation_info = AllocationCreateDesc {
            name: "Buffer",
            requirements: memory_requirements,
            location,
            linear: false,
        };

        let allocation = device
            .memory_allocator()
            .lock()
            .unwrap()
            .allocate(&allocation_info)?;

        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };

        Ok(Self {
            buffer_handle: buffer,
            allocation,
        })
    }

    /// See [`Buffer::new`]. The buffer size will be that needed to store the given data. The data will be
    /// stored in a GPU-side buffer via a staging buffer.
    ///
    /// # Errors
    ///
    /// This function can error if either the staging or gpu-side buffers fail to be created, or if copying the
    /// data from the staging buffer to the gpu-side buffer fails.
    pub fn new_with_data<T>(
        device: &Device,
        usage: ash::vk::BufferUsageFlags,
        data: &[T],
    ) -> Result<Self, Box<dyn Error>> {
        let data_size = (data.len() * std::mem::size_of::<T>()).try_into().unwrap();

        let mut staging_buffer: Buffer = Buffer::new(
            device,
            data_size,
            ash::vk::BufferUsageFlags::TRANSFER_SRC,
            // This memory location is the closest type to VMA's CPU-only memory.
            MemoryLocation::GpuToCpu,
        )?;

        unsafe {
            // No need to map first or unmap after, as gpu_allocator maps the memory when allocating.
            let memory_pointer =
                staging_buffer.allocation().mapped_ptr().unwrap().as_ptr() as *mut T;
            memory_pointer.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }

        let destination_buffer = Buffer::new(
            device,
            data_size,
            usage | ash::vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;

        unsafe {
            device.perform_immediate_submission(|command_buffer| {
                let buffer_region = ash::vk::BufferCopy::builder().size(data_size).build();

                device.cmd_copy_buffer(
                    command_buffer,
                    *staging_buffer.handle(),
                    *destination_buffer.handle(),
                    &[buffer_region],
                );

                Ok(())
            })?;
        }

        staging_buffer.destroy(device);

        Ok(destination_buffer)
    }

    pub fn handle(&self) -> &ash::vk::Buffer {
        &self.buffer_handle
    }

    pub fn allocation(&self) -> &Allocation {
        &self.allocation
    }
}

impl Destroy for Buffer {
    fn destroy(&mut self, device: &Device) {
        unsafe {
            device
                .memory_allocator()
                .lock()
                .unwrap()
                .free(std::mem::take(&mut self.allocation))
                .unwrap();

            device.destroy_buffer(self.buffer_handle, None);
        }
    }
}
