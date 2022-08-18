use std::error::Error;

use crate::*;

/// A swapchain with multiple images to render to, which can be presented to the screen.
pub struct Swapchain {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain_handle: ash::vk::SwapchainKHR,

    surface_format: ash::vk::SurfaceFormatKHR,
    extent: ash::vk::Extent2D,

    images: Vec<ash::vk::Image>,
    image_views: Vec<ash::vk::ImageView>,
}

impl Swapchain {
    /// Creates a swapchain with a supported surface format, present mode, and extent.
    /// `window_dimensions` is a tuple in the form (width: u32, height: u32).
    pub fn new(
        window_dimensions: (u32, u32),
        instance: &Instance,
        surface: &Surface,
        device: &Device,
        old_swapchain: Option<&Swapchain>,
    ) -> Result<Self, Box<dyn Error>> {
        let swapchain_loader =
            ash::extensions::khr::Swapchain::new(instance.handle(), device.handle());

        let supported_surface_formats = unsafe {
            surface.loader().get_physical_device_surface_formats(
                *device.physical_device_handle(),
                *surface.handle(),
            )?
        };

        // Use B8G8R8A8_SRGB with nonlinear SRGB colourspace if supported, or whatever supported format is first.
        let swapchain_image_format = supported_surface_formats
            .iter()
            .cloned()
            .find(|format| {
                format.format == ash::vk::Format::B8G8R8A8_SRGB
                    && format.color_space == ash::vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or_else(|| supported_surface_formats[0]);

        let supported_surface_capabilities = unsafe {
            surface.loader().get_physical_device_surface_capabilities(
                *device.physical_device_handle(),
                *surface.handle(),
            )?
        };

        // Use the whole window size if possible, or as large as is supported if not.
        let swapchain_extent = match supported_surface_capabilities.current_extent.width {
            std::u32::MAX => ash::vk::Extent2D::builder()
                .width(window_dimensions.0.clamp(
                    supported_surface_capabilities.min_image_extent.width,
                    supported_surface_capabilities.max_image_extent.width,
                ))
                .height(window_dimensions.1.clamp(
                    supported_surface_capabilities.min_image_extent.height,
                    supported_surface_capabilities.max_image_extent.height,
                ))
                .build(),
            _ => supported_surface_capabilities.current_extent,
        };

        let swapchain_image_count = if supported_surface_capabilities.max_image_count > 0
            && supported_surface_capabilities.min_image_count + 1
                > supported_surface_capabilities.max_image_count
        {
            supported_surface_capabilities.max_image_count
        } else {
            supported_surface_capabilities.min_image_count + 1
        };

        // Try to use the mailbox present mode, but fall back to fifo if mailbox is unsupported, as
        // fifo is guaranteed to be supported.
        let swapchain_present_mode = unsafe {
            surface
                .loader()
                .get_physical_device_surface_present_modes(
                    *device.physical_device_handle(),
                    *surface.handle(),
                )?
                .into_iter()
                .find(|&mode| mode == ash::vk::PresentModeKHR::MAILBOX)
                .unwrap_or(ash::vk::PresentModeKHR::FIFO)
        };

        let queue_family_indices = [device.graphics_family_index()];

        let swapchain_info = ash::vk::SwapchainCreateInfoKHR::builder()
            .flags(ash::vk::SwapchainCreateFlagsKHR::empty())
            .surface(*surface.handle())
            .min_image_count(swapchain_image_count)
            .image_format(swapchain_image_format.format)
            .image_color_space(swapchain_image_format.color_space)
            .image_extent(swapchain_extent)
            .image_array_layers(1)
            .image_usage(ash::vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(ash::vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(supported_surface_capabilities.current_transform)
            .composite_alpha(ash::vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(swapchain_present_mode)
            .clipped(true)
            .old_swapchain(
                old_swapchain.map_or(ash::vk::SwapchainKHR::null(), |swapchain| {
                    swapchain.swapchain_handle
                }),
            );

        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None)? };

        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };

        let mut swapchain_image_views: Vec<ash::vk::ImageView> = Vec::new();
        for swapchain_image in swapchain_images.iter() {
            let image_view_info = ash::vk::ImageViewCreateInfo::builder()
                .image(*swapchain_image)
                .view_type(ash::vk::ImageViewType::TYPE_2D)
                .format(swapchain_image_format.format)
                .components(
                    ash::vk::ComponentMapping::builder()
                        .r(ash::vk::ComponentSwizzle::IDENTITY)
                        .g(ash::vk::ComponentSwizzle::IDENTITY)
                        .b(ash::vk::ComponentSwizzle::IDENTITY)
                        .a(ash::vk::ComponentSwizzle::IDENTITY)
                        .build(),
                )
                .subresource_range(
                    ash::vk::ImageSubresourceRange::builder()
                        .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );

            let image_view = unsafe { device.handle().create_image_view(&image_view_info, None)? };

            swapchain_image_views.push(image_view);
        }

        log::debug!("Initialised swapchain.");

        Ok(Self {
            swapchain_loader,
            swapchain_handle: swapchain,
            surface_format: swapchain_image_format,
            extent: swapchain_extent,
            images: swapchain_images,
            image_views: swapchain_image_views,
        })
    }

    /// Get the images associated with the swapchain.
    pub fn get_images(&self) -> &Vec<ash::vk::Image> {
        &self.images
    }

    /// Get the view into each swapchain image.
    pub fn get_image_views(&self) -> &Vec<ash::vk::ImageView> {
        &self.image_views
    }
}

impl Destroy for Swapchain {
    fn destroy(&self, device: &Device) {
        unsafe {
            for &swapchain_image_view in self.image_views.iter() {
                device
                    .handle()
                    .destroy_image_view(swapchain_image_view, None);
            }

            self.swapchain_loader
                .destroy_swapchain(self.swapchain_handle, None);

            log::debug!("Destroyed swapchain");
        }
    }
}
