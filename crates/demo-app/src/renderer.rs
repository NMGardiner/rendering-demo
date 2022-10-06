use std::{
    error::Error,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use slab::Slab;

use winit::{
    event::{DeviceEvent, Event, WindowEvent},
    window::Window,
};

use nalgebra_glm as glm;

use crate::camera::*;

use rendering_engine::*;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Renderer {
    cursor_in_window: bool,

    camera: Camera,

    window_resize_flag: bool,
    should_render_flag: bool,

    current_time: f64,
    frame_index: usize,

    material_buffer: Buffer,

    // Descriptor stuff.
    descriptor_pool: ash::vk::DescriptorPool,
    sampler: ash::vk::Sampler,
    global_descriptor_set: ash::vk::DescriptorSet,
    texture_descriptor_set: ash::vk::DescriptorSet,

    test_mesh: Scene,

    textures: Slab<Image>,

    pipeline: Pipeline,
    depth_image: Image,
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
            .with_layers(&[
                #[cfg(debug_assertions)]
                "VK_LAYER_KHRONOS_validation",
            ])
            .with_extensions(&[
                ash::extensions::khr::GetPhysicalDeviceProperties2::name(),
                #[cfg(debug_assertions)]
                ash::extensions::ext::DebugUtils::name(),
            ])
            .build(Some(window))?;

        let surface = Surface::new(&window, &instance)?;

        let device = Device::builder()
            .with_core_features(
                ash::vk::PhysicalDeviceFeatures::builder()
                    .fill_mode_non_solid(true)
                    .build(),
            )
            .with_extensions(&[ash::extensions::khr::DynamicRendering::name()])
            .with_extension_feature(
                &mut ash::vk::PhysicalDeviceDynamicRenderingFeatures::builder()
                    .dynamic_rendering(true),
            )
            .with_extension_feature(
                &mut ash::vk::PhysicalDeviceDescriptorIndexingFeatures::builder()
                    .descriptor_binding_partially_bound(true)
                    .runtime_descriptor_array(true),
            )
            .build(&instance, Some(&surface))?;

        let swapchain = Swapchain::new(
            (window.inner_size().width, window.inner_size().height),
            &instance,
            &surface,
            &device,
            ash::vk::PresentModeKHR::FIFO,
            None,
        )?;

        let mut frames_in_flight: Vec<FrameResources> = vec![];
        for _i in 0..MAX_FRAMES_IN_FLIGHT {
            frames_in_flight.push(FrameResources::new(&device)?);
        }

        let depth_image = Image::new(
            &device,
            ash::vk::Extent3D::builder()
                .width(swapchain.extent().width)
                .height(swapchain.extent().height)
                .depth(1)
                .build(),
            ash::vk::Format::D32_SFLOAT,
            ash::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ash::vk::ImageAspectFlags::DEPTH,
            1,
        )?;

        let mut vert_shader =
            Shader::new(Path::new("./data/shaders/compiled/mesh.vert.spv"), &device)?;

        let mut frag_shader =
            Shader::new(Path::new("./data/shaders/compiled/mesh.frag.spv"), &device)?;

        let pipeline = Pipeline::builder()
            .shaders(&[&vert_shader, &frag_shader])
            .colour_formats(&[swapchain.surface_format().format])
            .depth_format(depth_image.format())
            .polygon_mode(ash::vk::PolygonMode::LINE)
            .build(&device)?;

        vert_shader.destroy(&device);
        frag_shader.destroy(&device);

        let mut textures = Slab::with_capacity(64);

        let mut test_mesh = Scene::load(
            &device,
            r"./data/assets/SimpleSkin/glTF/SimpleSkin.gltf",
            &mut textures,
        )?;

        test_mesh.update_joints();

        let descriptor_pool_sizes = [
            ash::vk::DescriptorPoolSize::builder()
                .ty(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1024)
                .build(),
            ash::vk::DescriptorPoolSize::builder()
                .ty(ash::vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(4)
                .build(),
        ];

        let descriptor_pool_info = ash::vk::DescriptorPoolCreateInfo::builder()
            .flags(ash::vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(8)
            .pool_sizes(&descriptor_pool_sizes);

        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None)? };

        let set_allocate_info = ash::vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(pipeline.descriptor_set_layouts());

        let descriptor_sets = unsafe { device.allocate_descriptor_sets(&set_allocate_info)? };

        let global_descriptor_set = descriptor_sets[0];
        let texture_descriptor_set = descriptor_sets[1];

        let material_buffer = Buffer::new_with_data(
            &device,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER,
            &test_mesh.materials,
        )?;

        let buffer_info = ash::vk::DescriptorBufferInfo::builder()
            .buffer(*material_buffer.handle())
            .offset(0)
            .range((std::mem::size_of::<MaterialData>() * test_mesh.materials.len()) as u64);

        let global_descriptor_write = ash::vk::WriteDescriptorSet::builder()
            .dst_binding(0)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
            .dst_set(global_descriptor_set)
            .dst_array_element(0)
            .buffer_info(std::slice::from_ref(&buffer_info));

        unsafe {
            device.update_descriptor_sets(std::slice::from_ref(&global_descriptor_write), &[]);
        }

        let sampler_info = ash::vk::SamplerCreateInfo::builder()
            .mag_filter(ash::vk::Filter::NEAREST)
            .min_filter(ash::vk::Filter::NEAREST)
            .address_mode_u(ash::vk::SamplerAddressMode::REPEAT)
            .address_mode_v(ash::vk::SamplerAddressMode::REPEAT)
            .address_mode_w(ash::vk::SamplerAddressMode::REPEAT);

        let sampler = unsafe { device.create_sampler(&sampler_info, None)? };

        let joint_buffer_info = ash::vk::DescriptorBufferInfo::builder()
            .buffer(*test_mesh.get_joint_buffer().handle())
            .offset(0)
            .range(test_mesh.get_joint_buffer().allocation().size());

        let joint_buffer_write = ash::vk::WriteDescriptorSet::builder()
            .dst_binding(1)
            .descriptor_type(ash::vk::DescriptorType::STORAGE_BUFFER)
            .dst_set(texture_descriptor_set)
            .dst_array_element(0)
            .buffer_info(std::slice::from_ref(&joint_buffer_info));

        unsafe {
            device.update_descriptor_sets(std::slice::from_ref(&joint_buffer_write), &[]);
        }

        if !textures.is_empty() {
            let image_infos = textures
                .iter()
                .map(|(_, texture)| {
                    ash::vk::DescriptorImageInfo::builder()
                        .image_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(*texture.view())
                        .sampler(sampler)
                        .build()
                })
                .collect::<Vec<_>>();

            let texture_descriptor_write = ash::vk::WriteDescriptorSet::builder()
                .dst_binding(0)
                .descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .dst_set(texture_descriptor_set)
                .dst_array_element(0)
                .image_info(&image_infos);

            unsafe {
                device.update_descriptor_sets(std::slice::from_ref(&texture_descriptor_write), &[]);
            }
        }

        let mut camera = Camera::new();
        camera.set_position(0.0, 0.0, 2.0);

        Ok(Renderer {
            cursor_in_window: false,
            camera,
            should_render_flag: true,
            window_resize_flag: false,
            current_time: 0.0,
            frame_index: 0,
            material_buffer,
            descriptor_pool,
            global_descriptor_set,
            texture_descriptor_set,
            sampler,
            test_mesh,
            textures,
            pipeline,
            depth_image,
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
                WindowEvent::KeyboardInput { input, .. } => {
                    if self.camera.enabled() {
                        if let winit::event::KeyboardInput {
                            state: winit::event::ElementState::Pressed,
                            virtual_keycode: Some(keycode),
                            ..
                        } = input
                        {
                            match keycode {
                                winit::event::VirtualKeyCode::W => {
                                    self.camera.translate(0.0, 0.0, 0.1);
                                }
                                winit::event::VirtualKeyCode::A => {
                                    self.camera.translate(-0.1, 0.0, 0.0);
                                }
                                winit::event::VirtualKeyCode::S => {
                                    self.camera.translate(0.0, 0.0, -0.1);
                                }
                                winit::event::VirtualKeyCode::D => {
                                    self.camera.translate(0.1, 0.0, 0.0);
                                }
                                winit::event::VirtualKeyCode::Escape => {
                                    self.camera.set_enabled(false);
                                    window.set_cursor_visible(true);
                                    window.set_cursor_grab(winit::window::CursorGrabMode::None)?;
                                }
                                _ => {}
                            }
                        }
                    }
                }
                WindowEvent::CursorEntered { .. } => {
                    self.cursor_in_window = true;
                }
                WindowEvent::CursorLeft { .. } => {
                    self.cursor_in_window = false;
                }
                _ => (),
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta, .. } => {
                    if self.camera.enabled() {
                        self.camera
                            .rotate((delta.0 / 8.0) as f32, (-delta.1 / 8.0) as f32);
                    }
                }
                DeviceEvent::Button {
                    button: 1,
                    state: winit::event::ElementState::Pressed,
                } => {
                    if self.cursor_in_window {
                        self.camera.set_enabled(true);

                        window.set_cursor_visible(false);

                        window
                            .set_cursor_grab(winit::window::CursorGrabMode::Confined)
                            .unwrap();
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
            .acquire_next_image(frame.image_acquired_semaphore());

        if acquire_result.is_err()
            && acquire_result.err().unwrap() == ash::vk::Result::ERROR_OUT_OF_DATE_KHR
        {
            self.recreate_swapchain(window)?;
            return Ok(());
        }

        let current_time = SystemTime::now().duration_since(UNIX_EPOCH)?;
        let delta = if self.current_time == 0.0 {
            0.0
        } else {
            current_time.as_secs_f64() - self.current_time
        };

        self.current_time = current_time.as_secs_f64();

        self.test_mesh.update_animations(delta);

        let image_index = acquire_result.unwrap().0;

        frame.begin_command_buffer(
            &self.device,
            ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        )?;

        let colour_subresource_range = ash::vk::ImageSubresourceRange::builder()
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
                .image(self.swapchain.images()[image_index as usize])
                .subresource_range(colour_subresource_range)
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
                .image(self.swapchain.images()[image_index as usize])
                .subresource_range(colour_subresource_range)
                .build(),
            ash::vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(
                    ash::vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | ash::vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .src_access_mask(ash::vk::AccessFlags2::empty())
                .dst_stage_mask(
                    ash::vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | ash::vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .dst_access_mask(ash::vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(ash::vk::ImageLayout::UNDEFINED)
                .new_layout(ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(ash::vk::QUEUE_FAMILY_IGNORED)
                .image(*self.depth_image.handle())
                .subresource_range(*self.depth_image.subresource_range())
                .build(),
        ];

        let dependency_info = ash::vk::DependencyInfo::builder()
            .dependency_flags(ash::vk::DependencyFlags::empty())
            .memory_barriers(&[])
            .buffer_memory_barriers(&[])
            .image_memory_barriers(&image_memory_barriers);

        unsafe {
            self.device
                .cmd_pipeline_barrier2(*frame.command_buffer(), &dependency_info);
        }

        let colour_attachment_rendering_info = ash::vk::RenderingAttachmentInfoKHR::builder()
            .image_view(self.swapchain.image_views()[image_index as usize])
            .image_layout(ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(ash::vk::ResolveModeFlags::NONE)
            .load_op(ash::vk::AttachmentLoadOp::CLEAR)
            .store_op(ash::vk::AttachmentStoreOp::STORE)
            .clear_value(ash::vk::ClearValue {
                color: ash::vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.1, 1.0],
                },
            });

        let depth_attachment_rendering_info = ash::vk::RenderingAttachmentInfoKHR::builder()
            .image_view(*self.depth_image.view())
            .image_layout(ash::vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .resolve_mode(ash::vk::ResolveModeFlags::NONE)
            .load_op(ash::vk::AttachmentLoadOp::CLEAR)
            .store_op(ash::vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(ash::vk::ClearValue {
                depth_stencil: ash::vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });

        let render_area = ash::vk::Rect2D::builder()
            .extent(self.swapchain.extent())
            .offset(*ash::vk::Offset2D::builder().x(0).y(0))
            .build();

        let rendering_info = ash::vk::RenderingInfoKHR::builder()
            .flags(ash::vk::RenderingFlagsKHR::empty())
            .render_area(render_area)
            .layer_count(1)
            .view_mask(0)
            .color_attachments(std::slice::from_ref(&colour_attachment_rendering_info))
            .depth_attachment(&depth_attachment_rendering_info);

        unsafe {
            self.device
                .cmd_begin_rendering(*frame.command_buffer(), &rendering_info);

            // Use a flipped (negative height) viewport.
            let viewport = ash::vk::Viewport::builder()
                .x(0.0)
                .y(self.swapchain.extent().height as f32)
                .width(self.swapchain.extent().width as f32)
                .height(-(self.swapchain.extent().height as f32))
                .min_depth(0.0)
                .max_depth(1.0);

            self.device.cmd_set_viewport(
                *frame.command_buffer(),
                0,
                std::slice::from_ref(&viewport),
            );

            let scissor = ash::vk::Rect2D::builder()
                .offset(ash::vk::Offset2D { x: 0, y: 0 })
                .extent(ash::vk::Extent2D {
                    width: self.swapchain.extent().width,
                    height: self.swapchain.extent().height,
                });

            self.device
                .cmd_set_scissor(*frame.command_buffer(), 0, std::slice::from_ref(&scissor));

            self.device.cmd_bind_pipeline(
                *frame.command_buffer(),
                ash::vk::PipelineBindPoint::GRAPHICS,
                *self.pipeline.handle(),
            );

            self.device.cmd_bind_descriptor_sets(
                *frame.command_buffer(),
                ash::vk::PipelineBindPoint::GRAPHICS,
                *self.pipeline.layout(),
                0,
                &[self.global_descriptor_set, self.texture_descriptor_set],
                &[],
            );

            self.device.cmd_bind_vertex_buffers(
                *frame.command_buffer(),
                0,
                &[*self.test_mesh.vertex_buffer.handle()],
                &[0],
            );

            if let Some(index_buffer) = &self.test_mesh.index_buffer {
                self.device.cmd_bind_index_buffer(
                    *frame.command_buffer(),
                    *index_buffer.handle(),
                    0,
                    ash::vk::IndexType::UINT32,
                );
            }

            let view_matrix = glm::look_at(
                &self.camera.get_position(),
                &(self.camera.get_position() + self.camera.get_front()),
                &glm::vec3(0.0, 1.0, 0.0),
            );

            let aspect =
                self.swapchain.extent().width as f32 / self.swapchain.extent().height as f32;

            let projection_matrix = glm::perspective_zo(aspect, 1.222, 0.1, 100.0);

            let pv_matrix = projection_matrix * view_matrix;

            self.test_mesh.draw(
                &self.device,
                frame.command_buffer(),
                self.pipeline.layout(),
                pv_matrix,
            );

            self.device.cmd_end_rendering(*frame.command_buffer());
        }

        frame.end_command_buffer(&self.device)?;

        let submit_info = ash::vk::SubmitInfo::builder()
            .wait_semaphores(std::slice::from_ref(frame.image_acquired_semaphore()))
            .wait_dst_stage_mask(&[ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(std::slice::from_ref(frame.command_buffer()))
            .signal_semaphores(std::slice::from_ref(frame.render_finished_semaphore()));

        let presentation_queue = self.device.graphics_queue();
        unsafe {
            self.device.queue_submit(
                *presentation_queue,
                std::slice::from_ref(&submit_info),
                *frame.render_finished_fence(),
            )?;
        }

        let swapchains = [*self.swapchain.handle()];

        let image_indices = [image_index];
        let semaphores = [*frame.render_finished_semaphore()];
        let present_info = ash::vk::PresentInfoKHR::builder()
            .wait_semaphores(&semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);

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
            self.swapchain.present_mode(),
            Some(&self.swapchain),
        )?;

        let old_swapchain = std::mem::replace(&mut self.swapchain, new_swapchain);

        // Push the old swapchain to this frame's deferred deletion queue, to be destroyed once we're sure
        // it's no longer in use.
        self.frames_in_flight[self.frame_index % MAX_FRAMES_IN_FLIGHT]
            .defer_object_deletion(old_swapchain);

        let new_depth_image = Image::new(
            &self.device,
            ash::vk::Extent3D::builder()
                .width(self.swapchain.extent().width)
                .height(self.swapchain.extent().height)
                .depth(1)
                .build(),
            ash::vk::Format::D32_SFLOAT,
            ash::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            ash::vk::ImageAspectFlags::DEPTH,
            1,
        )?;

        let old_depth_image = std::mem::replace(&mut self.depth_image, new_depth_image);

        self.frames_in_flight[self.frame_index % MAX_FRAMES_IN_FLIGHT]
            .defer_object_deletion(old_depth_image);

        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }

        // Destroy any objects that need manual destruction.
        // The rest of the renderer objects are destroyed in drop() after this.

        self.test_mesh.destroy(&self.device);

        self.material_buffer.destroy(&self.device);

        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }

        for (_, texture) in self.textures.iter_mut() {
            texture.destroy(&self.device);
        }

        self.pipeline.destroy(&self.device);

        self.depth_image.destroy(&self.device);

        for frame in self.frames_in_flight.iter_mut() {
            frame.destroy(&self.device);
        }

        self.swapchain.destroy(&self.device);
    }
}
