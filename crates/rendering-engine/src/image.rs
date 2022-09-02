use std::error::Error;

use crate::*;

use gpu_allocator::MemoryLocation;

/// A Vulkan image object, allocated using gpu_allocator.
pub struct Image {
    image_handle: ash::vk::Image,
    allocation: gpu_allocator::vulkan::Allocation,

    view: ash::vk::ImageView,
    subresource_range: ash::vk::ImageSubresourceRange,
    format: ash::vk::Format,
}

impl Image {
    /// Create a new image with the given parameters. The allocated memory will be mapped and bound, and
    /// a 2D image view created.
    ///
    /// # Errors
    ///
    /// This function can error if `ash` fails to bind the image memory, create the image/view, or if
    /// `gpu_allocator` fails to allocate the required memory.
    pub fn new(
        device: &Device,
        extent: ash::vk::Extent3D,
        format: ash::vk::Format,
        usage: ash::vk::ImageUsageFlags,
        aspect_mask: ash::vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let image_info = ash::vk::ImageCreateInfo::builder()
            .flags(ash::vk::ImageCreateFlags::empty())
            .image_type(ash::vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            .samples(ash::vk::SampleCountFlags::TYPE_1)
            .tiling(ash::vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
            .initial_layout(ash::vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_info, None)? };

        let memory_requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation_info = gpu_allocator::vulkan::AllocationCreateDesc {
            name: "Image",
            requirements: memory_requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
        };

        let allocation = device
            .memory_allocator()
            .lock()
            .unwrap()
            .allocate(&allocation_info)?;

        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? }

        let subresource_range = ash::vk::ImageSubresourceRange::builder()
            .aspect_mask(aspect_mask)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        let image_view_info = ash::vk::ImageViewCreateInfo::builder()
            .flags(ash::vk::ImageViewCreateFlags::empty())
            .image(image)
            .view_type(ash::vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range);

        let image_view = unsafe { device.create_image_view(&image_view_info, None)? };

        Ok(Self {
            image_handle: image,
            allocation,
            view: image_view,
            subresource_range,
            format,
        })
    }

    /// See [`Image::new`]. The image size will be that needed to store the given data. The data will be
    /// stored in a GPU-side image via a staging buffer.
    pub fn new_with_data<T>(
        device: &Device,
        data: &[T],
        extent: ash::vk::Extent3D,
        format: ash::vk::Format,
        usage: ash::vk::ImageUsageFlags,
        aspect_mask: ash::vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> Result<Self, Box<dyn Error>> {
        let mut staging_buffer: Buffer = Buffer::new(
            device,
            (data.len() * std::mem::size_of::<T>()) as u64,
            ash::vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuToCpu,
        )?;

        unsafe {
            let memory_pointer =
                staging_buffer.allocation().mapped_ptr().unwrap().as_ptr() as *mut T;
            memory_pointer.copy_from_nonoverlapping(data.as_ptr(), data.len());
        }

        let image = Image::new(
            device,
            extent,
            format,
            usage | ash::vk::ImageUsageFlags::TRANSFER_DST,
            aspect_mask,
            mip_levels,
        )?;

        device.perform_immediate_submission(|command_buffer| {
            // Transition the image for writing.
            device.perform_image_layout_transition(
                &command_buffer,
                image.handle(),
                ash::vk::PipelineStageFlags2::TOP_OF_PIPE,
                ash::vk::AccessFlags2::empty(),
                ash::vk::PipelineStageFlags2::TRANSFER,
                ash::vk::AccessFlags2::TRANSFER_WRITE,
                ash::vk::ImageLayout::UNDEFINED,
                ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image.subresource_range(),
            )?;

            let buffer_region = ash::vk::BufferImageCopy::builder()
                .buffer_row_length(0)
                .buffer_image_height(0)
                .buffer_offset(0)
                .image_subresource(
                    ash::vk::ImageSubresourceLayers::builder()
                        .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
                        .mip_level(0)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .image_extent(extent);

            // Write the data in the staging buffer to the image.
            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    *staging_buffer.handle(),
                    *image.handle(),
                    ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[*buffer_region],
                );
            }

            // Transition the image for reading.
            device.perform_image_layout_transition(
                &command_buffer,
                image.handle(),
                ash::vk::PipelineStageFlags2::TRANSFER,
                ash::vk::AccessFlags2::TRANSFER_WRITE,
                ash::vk::PipelineStageFlags2::FRAGMENT_SHADER,
                ash::vk::AccessFlags2::SHADER_READ,
                ash::vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image.subresource_range(),
            )
        })?;

        staging_buffer.destroy(device);

        Ok(image)
    }

    pub fn handle(&self) -> &ash::vk::Image {
        &self.image_handle
    }

    pub fn view(&self) -> &ash::vk::ImageView {
        &self.view
    }

    pub fn subresource_range(&self) -> &ash::vk::ImageSubresourceRange {
        &self.subresource_range
    }

    pub fn format(&self) -> ash::vk::Format {
        self.format
    }
}

impl Destroy for Image {
    fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_image_view(self.view, None);

            device
                .memory_allocator()
                .lock()
                .unwrap()
                .free(std::mem::take(&mut self.allocation))
                .unwrap();

            device.destroy_image(self.image_handle, None);
        }
    }
}
