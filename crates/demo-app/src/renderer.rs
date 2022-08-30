use std::{error::Error, path::Path};

use winit::{
    event::{DeviceEvent, Event, WindowEvent},
    window::Window,
};

use nalgebra_glm as glm;

use crate::camera::*;

use rendering_engine::*;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Renderer {
    texture_index: u32,

    camera: Camera,

    window_resize_flag: bool,
    should_render_flag: bool,

    frame_index: usize,

    // glTF models may not provide indices.
    index_buffer: Option<Buffer>,
    vertex_buffer: Buffer,

    // Descriptor stuff.
    descriptor_pool: ash::vk::DescriptorPool,
    sampler: ash::vk::Sampler,
    descriptor_set: ash::vk::DescriptorSet,

    model_data: ModelData,
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
            .build(&device)?;

        vert_shader.destroy(&device);
        frag_shader.destroy(&device);

        let model_data = load_model(&device)?;

        let vertex_buffer = Buffer::new_with_data(
            &device,
            ash::vk::BufferUsageFlags::VERTEX_BUFFER,
            model_data.vertices.clone(),
        )?;

        let index_buffer = if model_data.indices.is_empty() {
            None
        } else {
            Some(Buffer::new_with_data(
                &device,
                ash::vk::BufferUsageFlags::INDEX_BUFFER,
                model_data.indices.clone(),
            )?)
        };

        let descriptor_pool_sizes = [ash::vk::DescriptorPoolSize::builder()
            .ty(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1024)
            .build()];

        let descriptor_pool_info = ash::vk::DescriptorPoolCreateInfo::builder()
            .flags(ash::vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&descriptor_pool_sizes);

        let descriptor_pool = unsafe {
            device
                .handle()
                .create_descriptor_pool(&descriptor_pool_info, None)?
        };

        let set_allocate_info = ash::vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(pipeline.descriptor_set_layouts());

        let texture_descriptor_set = unsafe {
            device
                .handle()
                .allocate_descriptor_sets(&set_allocate_info)?
        }[0];

        let sampler_info = ash::vk::SamplerCreateInfo::builder()
            .mag_filter(ash::vk::Filter::NEAREST)
            .min_filter(ash::vk::Filter::NEAREST)
            .address_mode_u(ash::vk::SamplerAddressMode::REPEAT)
            .address_mode_v(ash::vk::SamplerAddressMode::REPEAT)
            .address_mode_w(ash::vk::SamplerAddressMode::REPEAT);

        let sampler = unsafe { device.handle().create_sampler(&sampler_info, None)? };

        let image_infos = model_data
            .textures
            .iter()
            .map(|texture| {
                ash::vk::DescriptorImageInfo::builder()
                    .image_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(*texture.view())
                    .sampler(sampler)
                    .build()
            })
            .collect::<Vec<_>>();

        let descriptor_write = ash::vk::WriteDescriptorSet::builder()
            .dst_binding(0)
            .descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .dst_set(texture_descriptor_set)
            .dst_array_element(0)
            .image_info(&image_infos)
            .build();

        unsafe {
            device
                .handle()
                .update_descriptor_sets(&[descriptor_write], &[]);
        }

        let mut camera = Camera::new();
        camera.set_position(0.0, 0.0, 2.0);

        Ok(Renderer {
            texture_index: 0,
            camera,
            should_render_flag: true,
            window_resize_flag: false,
            frame_index: 0,
            index_buffer,
            vertex_buffer,
            descriptor_pool,
            descriptor_set: texture_descriptor_set,
            sampler,
            model_data,
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
                            winit::event::VirtualKeyCode::E => {
                                // Cycle through the model's textures.
                                self.texture_index = (self.texture_index + 1)
                                    % self.model_data.textures.len() as u32;
                            }
                            _ => {}
                        }
                    }
                }
                _ => (),
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta, .. } => {
                    self.camera
                        .rotate((delta.0 / 8.0) as f32, (-delta.1 / 8.0) as f32);
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
                .handle()
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
            })
            .build();

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
            })
            .build();

        let render_area = ash::vk::Rect2D::builder()
            .extent(self.swapchain.extent())
            .offset(*ash::vk::Offset2D::builder().x(0).y(0))
            .build();

        let rendering_info = ash::vk::RenderingInfoKHR::builder()
            .flags(ash::vk::RenderingFlagsKHR::empty())
            .render_area(render_area)
            .layer_count(1)
            .view_mask(0)
            .color_attachments(&[colour_attachment_rendering_info])
            .depth_attachment(&depth_attachment_rendering_info)
            .build();

        unsafe {
            self.device
                .handle()
                .cmd_begin_rendering(*frame.command_buffer(), &rendering_info);

            // Use a flipped (negative height) viewport.
            let viewport = ash::vk::Viewport::builder()
                .x(0.0)
                .y(self.swapchain.extent().height as f32)
                .width(self.swapchain.extent().width as f32)
                .height(-(self.swapchain.extent().height as f32))
                .min_depth(0.0)
                .max_depth(1.0)
                .build();

            self.device
                .handle()
                .cmd_set_viewport(*frame.command_buffer(), 0, &[viewport]);

            let scissor = ash::vk::Rect2D::builder()
                .offset(ash::vk::Offset2D { x: 0, y: 0 })
                .extent(ash::vk::Extent2D {
                    width: self.swapchain.extent().width,
                    height: self.swapchain.extent().height,
                })
                .build();

            self.device
                .handle()
                .cmd_set_scissor(*frame.command_buffer(), 0, &[scissor]);

            self.device.handle().cmd_bind_pipeline(
                *frame.command_buffer(),
                ash::vk::PipelineBindPoint::GRAPHICS,
                *self.pipeline.handle(),
            );

            self.device.handle().cmd_bind_descriptor_sets(
                *frame.command_buffer(),
                ash::vk::PipelineBindPoint::GRAPHICS,
                *self.pipeline.layout(),
                0,
                &[self.descriptor_set],
                &[],
            );

            self.device.handle().cmd_bind_vertex_buffers(
                *frame.command_buffer(),
                0,
                &[*self.vertex_buffer.handle()],
                &[0],
            );

            if let Some(index_buffer) = &self.index_buffer {
                self.device.handle().cmd_bind_index_buffer(
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

            for (index, vertex_offset) in self.model_data.vertex_offsets.iter().enumerate() {
                if let Some(matrix) = self.model_data.matrices[index] {
                    let push_constants = PushConstants {
                        matrix: projection_matrix * view_matrix * matrix,
                    };

                    let push_constants_slice = std::slice::from_raw_parts(
                        &push_constants as *const PushConstants as *const u8,
                        std::mem::size_of::<PushConstants>() / std::mem::size_of::<u8>(),
                    );

                    self.device.handle().cmd_push_constants(
                        *frame.command_buffer(),
                        *self.pipeline.layout(),
                        ash::vk::ShaderStageFlags::VERTEX,
                        0,
                        push_constants_slice,
                    );
                }

                if self.index_buffer.is_some() {
                    let index_count = if self.model_data.index_offsets.len() > index + 1 {
                        self.model_data.index_offsets[index + 1]
                            - self.model_data.index_offsets[index]
                    } else {
                        self.model_data.indices.len() - self.model_data.index_offsets[index]
                    };

                    self.device.handle().cmd_draw_indexed(
                        *frame.command_buffer(),
                        index_count as u32,
                        1,
                        self.model_data.index_offsets[index] as u32,
                        *vertex_offset as i32,
                        self.texture_index,
                    );
                } else {
                    let vertex_count = if self.model_data.vertex_offsets.len() > index + 1 {
                        self.model_data.vertex_offsets[index + 1]
                            - self.model_data.vertex_offsets[index]
                    } else {
                        self.model_data.vertices.len() - self.model_data.vertex_offsets[index]
                    };

                    self.device.handle().cmd_draw(
                        *frame.command_buffer(),
                        vertex_count as u32,
                        1,
                        self.model_data.vertex_offsets[index] as u32,
                        self.texture_index,
                    );
                }
            }

            self.device
                .handle()
                .cmd_end_rendering(*frame.command_buffer());
        }

        frame.end_command_buffer(&self.device)?;

        let submit_info = ash::vk::SubmitInfo::builder()
            .wait_semaphores(&[*frame.image_acquired_semaphore()])
            .wait_dst_stage_mask(&[ash::vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
            .command_buffers(&[*frame.command_buffer()])
            .signal_semaphores(&[*frame.render_finished_semaphore()])
            .build();

        let presentation_queue = self.device.graphics_queue();
        unsafe {
            self.device.handle().queue_submit(
                *presentation_queue,
                &[submit_info],
                *frame.render_finished_fence(),
            )?;
        }

        let swapchains = [*self.swapchain.handle()];

        let image_indices = [image_index];
        let present_info = ash::vk::PresentInfoKHR::builder()
            .wait_semaphores(&[*frame.render_finished_semaphore()])
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
            self.swapchain.present_mode(),
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

        unsafe {
            self.device.handle().destroy_sampler(self.sampler, None);
            self.device
                .handle()
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }

        for texture in self.model_data.textures.iter_mut() {
            texture.destroy(&self.device);
        }

        if let Some(index_buffer) = &mut self.index_buffer {
            index_buffer.destroy(&self.device);
        }
        self.vertex_buffer.destroy(&self.device);

        self.pipeline.destroy(&self.device);

        self.depth_image.destroy(&self.device);

        for frame in self.frames_in_flight.iter_mut() {
            frame.destroy(&self.device);
        }

        self.swapchain.destroy(&self.device);
    }
}

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: glm::Vec3,
    pub colour: glm::Vec3,
    pub normal: glm::Vec3,
    pub texture_coords: glm::Vec2,
}

pub struct PushConstants {
    pub matrix: glm::Mat4,
}

pub struct ModelData {
    vertices: Vec<Vertex>,
    vertex_offsets: Vec<usize>,
    indices: Vec<u32>,
    index_offsets: Vec<usize>,
    matrices: Vec<Option<glm::Mat4>>,
    textures: Vec<Image>,
}

pub fn load_model(device: &Device) -> Result<ModelData, Box<dyn Error>> {
    let (document, buffers, images) =
        gltf::import(Path::new(r"./data/assets/DamagedHelmet/DamagedHelmet.gltf"))?;

    let mut vertices: Vec<Vertex> = vec![];
    let mut indices: Vec<u32> = vec![];
    let mut vertex_offsets: Vec<usize> = vec![];
    let mut index_offsets: Vec<usize> = vec![];
    let mut matrices: Vec<Option<glm::Mat4>> = vec![];
    let mut textures: Vec<Image> = vec![];

    for image in images.iter() {
        let format = match image.format {
            gltf::image::Format::R8 => ash::vk::Format::R8_SRGB,
            gltf::image::Format::R8G8 => ash::vk::Format::R8G8_SRGB,
            gltf::image::Format::R8G8B8 => ash::vk::Format::R8G8B8_SRGB,
            gltf::image::Format::R8G8B8A8 => ash::vk::Format::R8G8B8A8_SRGB,
            gltf::image::Format::B8G8R8 => ash::vk::Format::B8G8R8_SRGB,
            gltf::image::Format::B8G8R8A8 => ash::vk::Format::B8G8R8A8_SRGB,
            _ => {
                log::error!(
                    "This model uses an unsupported texture format: {:?}.",
                    image.format
                );
                ash::vk::Format::UNDEFINED
            }
        };

        textures.push(Image::new_with_data(
            device,
            &image.pixels,
            ash::vk::Extent3D::builder()
                .width(image.width)
                .height(image.height)
                .depth(1)
                .build(),
            format,
            ash::vk::ImageUsageFlags::SAMPLED,
            ash::vk::ImageAspectFlags::COLOR,
            1,
        )?);
    }

    for node in document.nodes() {
        let transform_matrix = node.transform().matrix();
        let transform_matrix: &[f32] = unsafe {
            std::slice::from_raw_parts(transform_matrix.as_ptr().cast(), transform_matrix.len() * 4)
        };

        let mut pushed = false;
        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                // Store the transformation matrix for each mesh. Pad with None to line up with vertex_offsets.
                if !pushed {
                    matrices.push(Some(glm::make_mat4(transform_matrix)));
                    pushed = true;
                } else {
                    matrices.push(None);
                }

                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let mut positions: Vec<glm::Vec3> = vec![];
                let mut normals: Vec<glm::Vec3> = vec![];
                let mut texture_coords: Vec<glm::Vec2> = vec![];

                if let Some(primitive_positions) = reader.read_positions() {
                    for primitive_position in primitive_positions {
                        positions.push(glm::make_vec3(&primitive_position));

                        // Push a default normal in case there's none given.
                        normals.push(glm::Vec3::zeros());

                        texture_coords.push(glm::Vec2::zeros());
                    }
                }

                if let Some(primitive_normals) = reader.read_normals() {
                    for (index, primitive_normal) in primitive_normals.enumerate() {
                        normals[index] = glm::make_vec3(&primitive_normal);
                    }
                }

                if let Some(primitive_texture_coords) = reader.read_tex_coords(0) {
                    match primitive_texture_coords {
                        gltf::mesh::util::ReadTexCoords::U8(data) => {
                            for (index, texture_coord) in data.enumerate() {
                                texture_coords[index] =
                                    glm::vec2(texture_coord[0] as f32, texture_coord[1] as f32);
                            }
                        }
                        gltf::mesh::util::ReadTexCoords::U16(data) => {
                            for (index, texture_coord) in data.enumerate() {
                                texture_coords[index] =
                                    glm::vec2(texture_coord[0] as f32, texture_coord[1] as f32);
                            }
                        }
                        gltf::mesh::util::ReadTexCoords::F32(data) => {
                            for (index, texture_coord) in data.enumerate() {
                                texture_coords[index] =
                                    glm::vec2(texture_coord[0], texture_coord[1]);
                            }
                        }
                    }
                }

                vertex_offsets.push(vertices.len());
                for (index, position) in positions.iter().enumerate() {
                    vertices.push(Vertex {
                        position: *position,
                        colour: glm::vec3(0.5, 0.5, 0.5),
                        normal: normals[index],
                        texture_coords: texture_coords[index],
                    });
                }

                index_offsets.push(indices.len());
                if let Some(primitive_indices) = reader.read_indices() {
                    match primitive_indices {
                        gltf::mesh::util::ReadIndices::U8(i) => {
                            indices.extend(i.map(|index| index as u32));
                        }
                        gltf::mesh::util::ReadIndices::U16(i) => {
                            indices.extend(i.map(|index| index as u32));
                        }
                        gltf::mesh::util::ReadIndices::U32(i) => {
                            indices.extend(i);
                        }
                    }
                }
            }
        };
    }

    Ok(ModelData {
        vertices,
        vertex_offsets,
        indices,
        index_offsets,
        matrices,
        textures,
    })
}
