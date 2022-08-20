use std::error::Error;

use winit::{
    event::{Event, WindowEvent},
    window::Window,
};

use rendering_engine::*;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Renderer {
    window_resize_flag: bool,
    should_render_flag: bool,

    frame_index: usize,

    frames_in_flight: Vec<FrameResources>,
    swapchain: Swapchain,
    device: Device,
    surface: Surface,
    instance: Instance,
}

impl Renderer {
    pub fn new(window: &Window) -> Result<Self, Box<dyn Error>> {
        let instance = Instance::builder()
            .application_name("Rendering Demo")
            .application_version(0, 1, 0)
            .window_handle(&window)
            .enable_validation_layers(cfg!(debug_assertions))
            .build()?;

        let surface = Surface::new(&window, &instance)?;

        let device = Device::new(&instance, Some(&surface))?;

        let swapchain = Swapchain::new(
            (window.inner_size().width, window.inner_size().height),
            &instance,
            &surface,
            &device,
            None,
        )?;

        let mut frames_in_flight: Vec<FrameResources> = vec![];
        for _i in 0..MAX_FRAMES_IN_FLIGHT {
            frames_in_flight.push(FrameResources::new(&device)?);
        }

        Ok(Renderer {
            should_render_flag: true,
            window_resize_flag: false,
            frame_index: 0,
            frames_in_flight,
            swapchain,
            device,
            surface,
            instance,
        })
    }

    pub fn handle_event(
        &mut self,
        event: &Event<()>,
        window: &Window,
    ) -> Result<(), Box<dyn Error>> {
        match event {
            Event::MainEventsCleared => {
                if self.should_render_flag {
                    self.draw_frame(window)?
                }
            }
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    self.should_render_flag = false;
                }
                WindowEvent::Resized(size) => {
                    if size.width == 0 || size.height == 0 {
                        self.should_render_flag = false;
                    } else {
                        self.should_render_flag = true;
                        self.window_resize_flag = true;
                    }
                }
                _ => (),
            },
            _ => (),
        }

        Ok(())
    }

    fn draw_frame(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        let frame_index = self.frame_index % MAX_FRAMES_IN_FLIGHT;

        let frame = &mut self.frames_in_flight[frame_index];

        frame.await_render_finished_fence(&self.device)?;
        frame.reset_render_finished_fence(&self.device)?;

        // Destroy any objects that had their deletion deferred in the previous use of this frame.
        frame.process_deferred_deletion_queue(&self.device);

        frame.reset_command_buffer(&self.device)?;

        let acquire_result = self
            .swapchain
            .acquire_next_image(frame.get_image_acquired_semaphore());

        if acquire_result.is_err()
            && acquire_result.err().unwrap() == ash::vk::Result::ERROR_OUT_OF_DATE_KHR
        {
            self.recreate_swapchain(window)?;
            return Ok(());
        }

        let image_index = acquire_result.unwrap().0;

        frame.begin_command_buffer(
            &self.device,
            ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        )?;

        let default_subresource_range = ash::vk::ImageSubresourceRange::builder()
            .aspect_mask(ash::vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        let image_memory_barriers = [
            ash::vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(ash::vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(ash::vk::AccessFlags2::empty())
                .dst_stage_mask(ash::vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(
                    ash::vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                        | ash::vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                )
                .old_layout(ash::vk::ImageLayout::UNDEFINED)
                .new_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .image(self.swapchain.get_images()[image_index as usize])
                .subresource_range(default_subresource_range)
                .build(),
            ash::vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(ash::vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(ash::vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(ash::vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
                .dst_access_mask(ash::vk::AccessFlags2::empty())
                .old_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(ash::vk::ImageLayout::PRESENT_SRC_KHR)
                .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .image(self.swapchain.get_images()[image_index as usize])
                .subresource_range(default_subresource_range)
                .build(),
        ];

        let dependency_info = ash::vk::DependencyInfo::builder()
            .dependency_flags(ash::vk::DependencyFlags::empty())
            .memory_barriers(&[])
            .buffer_memory_barriers(&[])
            .image_memory_barriers(&image_memory_barriers);

        unsafe {
            self.device
                .handle()
                .cmd_pipeline_barrier2(*frame.get_command_buffer(), &dependency_info);
        }

        let colour_attachment_rendering_info = ash::vk::RenderingAttachmentInfoKHR::builder()
            .image_view(self.swapchain.get_image_views()[image_index as usize])
            .image_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(ash::vk::ResolveModeFlags::NONE)
            .load_op(ash::vk::AttachmentLoadOp::CLEAR)
            .store_op(ash::vk::AttachmentStoreOp::STORE)
            .clear_value(ash::vk::ClearValue {
                color: ash::vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.1, 1.0],
                },
            })
            .build();

        let render_area = ash::vk::Rect2D::builder()
            .extent(self.swapchain.get_extent())
            .offset(*ash::vk::Offset2D::builder().x(0).y(0))
            .build();

        let rendering_info = ash::vk::RenderingInfoKHR::builder()
            .flags(ash::vk::RenderingFlagsKHR::empty())
            .render_area(render_area)
            .layer_count(1)
            .view_mask(0)
            .color_attachments(&[colour_attachment_rendering_info])
            .build();

        unsafe {
            self.device
                .handle()
                .cmd_begin_rendering(*frame.get_command_buffer(), &rendering_info);

            self.device
                .handle()
                .cmd_end_rendering(*frame.get_command_buffer());
        }

        frame.end_command_buffer(&self.device)?;

        let submit_info = ash::vk::SubmitInfo::builder()
            .wait_semaphores(&[*frame.get_image_acquired_semaphore()])
            .wait_dst_stage_mask(&[ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(&[*frame.get_command_buffer()])
            .signal_semaphores(&[*frame.get_render_finished_semaphore()])
            .build();

        let presentation_queue = self.device.graphics_queue();
        unsafe {
            self.device.handle().queue_submit(
                *presentation_queue,
                &[submit_info],
                *frame.get_render_finished_fence(),
            )?;
        }

        let swapchains = [*self.swapchain.handle()];

        let image_indices = [image_index];
        let present_info = ash::vk::PresentInfoKHR::builder()
            .wait_semaphores(&[*frame.get_render_finished_semaphore()])
            .swapchains(&swapchains)
            .image_indices(&image_indices)
            .build();

        let presentation_result = self
            .swapchain
            .queue_present(self.device.graphics_queue(), &present_info);

        // Recreate the swapchain if presentation returns ERROR_OUT_OF_DATE_KHR or if the swapchain is suboptimal.
        let should_recreate_swapchain = if let Some(error) = presentation_result.err() {
            error == ash::vk::Result::ERROR_OUT_OF_DATE_KHR
        } else {
            presentation_result.unwrap()
        };

        if should_recreate_swapchain || self.window_resize_flag {
            self.window_resize_flag = false;
            self.recreate_swapchain(window)?;
        }

        self.frame_index += 1;

        Ok(())
    }

    fn recreate_swapchain(&mut self, window: &Window) -> Result<(), Box<dyn Error>> {
        let new_swapchain = Swapchain::new(
            (window.inner_size().width, window.inner_size().height),
            &self.instance,
            &self.surface,
            &self.device,
            Some(&self.swapchain),
        )?;

        let old_swapchain = std::mem::replace(&mut self.swapchain, new_swapchain);

        // Push the old swapchain to this frame's deferred deletion queue, to be destroyed once we're sure
        // it's no longer in use.
        self.frames_in_flight[self.frame_index % MAX_FRAMES_IN_FLIGHT]
            .defer_object_deletion(old_swapchain);

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.handle().device_wait_idle().unwrap();
        }

        // Destroy any objects that need manual destruction.
        // The rest of the renderer objects are destroyed in drop() after this.
        for frame in self.frames_in_flight.iter() {
            frame.destroy(&self.device);
        }

        self.swapchain.destroy(&self.device);
    }
}
