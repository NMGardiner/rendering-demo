use std::error::Error;

use crate::*;

/// An object for building graphics pipelines.
pub struct PipelineBuilder<'a> {
    shaders: &'a [&'a Shader],
    topology: ash::vk::PrimitiveTopology,
    depth_clamp_enable: bool,
    polygon_mode: ash::vk::PolygonMode,
    line_width: f32,
    cull_mode: ash::vk::CullModeFlags,
    front_face: ash::vk::FrontFace,
    colour_formats: &'a [ash::vk::Format],
    depth_format: ash::vk::Format,
    stencil_format: ash::vk::Format,
}

impl<'a> PipelineBuilder<'a> {
    /// Build the graphics pipeline. This does not consume the builder, to allow for reuse.
    ///
    /// See [`Pipeline::new`] for details.
    pub fn build(&self, device: &Device) -> Result<Pipeline, Box<dyn Error>> {
        Pipeline::new(self, device)
    }

    /// Set the shaders to be used by the pipeline (defaults to empty).
    pub fn shaders(mut self, shaders: &'a [&'a Shader]) -> Self {
        self.shaders = shaders;
        self
    }

    /// Set the PrimitiveTopology (defaults to TRIANGLE_LIST).
    pub fn topology(mut self, topology: ash::vk::PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    /// Set whether to clamp fragments between the min and max depth (defaults to false).
    pub fn depth_clamp_enable(mut self, enable: bool) -> Self {
        self.depth_clamp_enable = enable;
        self
    }

    /// Set the PolygonMode (defaults to FILL).
    pub fn polygon_mode(mut self, polygon_mode: ash::vk::PolygonMode) -> Self {
        self.polygon_mode = polygon_mode;
        self
    }

    /// Set the line width (defaults to 1.0).
    pub fn line_width(mut self, line_width: f32) -> Self {
        self.line_width = line_width;
        self
    }

    /// Set the CullModeFlags (defaults to NONE).
    pub fn cull_mode(mut self, cull_mode: ash::vk::CullModeFlags) -> Self {
        self.cull_mode = cull_mode;
        self
    }

    /// Set the FrontFace (defaults to CLOCKWISE).
    pub fn front_face(mut self, front_face: ash::vk::FrontFace) -> Self {
        self.front_face = front_face;
        self
    }

    /// Set the colour format(s) (defaults to empty).
    pub fn colour_formats(mut self, formats: &'a [ash::vk::Format]) -> Self {
        self.colour_formats = formats;
        self
    }

    /// Set the depth format (defaults to UNDEFINED).
    pub fn depth_format(mut self, format: ash::vk::Format) -> Self {
        self.depth_format = format;
        self
    }

    /// Set the stencil format (defaults to UNDEFINED).
    pub fn stencil_format(mut self, format: ash::vk::Format) -> Self {
        self.stencil_format = format;
        self
    }
}

impl<'a> Default for PipelineBuilder<'a> {
    fn default() -> Self {
        Self {
            shaders: &[],
            topology: ash::vk::PrimitiveTopology::TRIANGLE_LIST,
            depth_clamp_enable: false,
            polygon_mode: ash::vk::PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: ash::vk::CullModeFlags::NONE,
            front_face: ash::vk::FrontFace::CLOCKWISE,
            colour_formats: &[],
            depth_format: ash::vk::Format::default(),
            stencil_format: ash::vk::Format::default(),
        }
    }
}

/// A graphics pipeline, storing the `ash` pipeline and pipeline layout.
pub struct Pipeline {
    pipeline_handle: ash::vk::Pipeline,
    pipeline_layout: ash::vk::PipelineLayout,
}

impl Pipeline {
    /// Create a [`PipelineBuilder`] to build a graphics pipeline.
    pub fn builder() -> PipelineBuilder<'static> {
        PipelineBuilder::default()
    }

    /// Create a new graphics pipeline from the given builder's parameters.
    ///
    /// # Errors
    ///
    /// This function can error if `ash` fails to create the pipeline or pipeline layout.
    pub fn new(builder: &PipelineBuilder, device: &Device) -> Result<Self, Box<dyn Error>> {
        let mut shader_stage_infos = vec![];
        let mut vertex_binding_descriptions: Vec<ash::vk::VertexInputBindingDescription> = vec![];
        let mut vertex_attribute_descriptions: Vec<ash::vk::VertexInputAttributeDescription> =
            vec![];
        let mut push_constant_ranges: Vec<ash::vk::PushConstantRange> = vec![];

        for shader in builder.shaders.iter() {
            shader_stage_infos.push(*shader.stage_info());
            vertex_binding_descriptions.extend(shader.binding_descriptions().clone());
            vertex_attribute_descriptions.extend(shader.attribute_descriptions().clone());
            push_constant_ranges.extend(shader.push_constant_ranges().clone());
        }

        let input_state_info = ash::vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descriptions)
            .vertex_attribute_descriptions(&vertex_attribute_descriptions);

        let input_assembly_state_info = ash::vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(builder.topology)
            .primitive_restart_enable(false);

        let viewport_state_info = ash::vk::PipelineViewportStateCreateInfo::default();

        let dynamic_state_info =
            ash::vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&[
                ash::vk::DynamicState::VIEWPORT_WITH_COUNT,
                ash::vk::DynamicState::SCISSOR_WITH_COUNT,
            ]);

        let rasterisation_state_info = ash::vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(builder.depth_clamp_enable)
            .rasterizer_discard_enable(false)
            .polygon_mode(builder.polygon_mode)
            .line_width(builder.line_width)
            .cull_mode(builder.cull_mode)
            .front_face(builder.front_face)
            .depth_bias_enable(false);

        let multisample_state_info = ash::vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(ash::vk::SampleCountFlags::TYPE_1);

        // TODO: Configurable per-attachment blend state.
        let attachment_states = [ash::vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                ash::vk::ColorComponentFlags::R
                    | ash::vk::ColorComponentFlags::G
                    | ash::vk::ColorComponentFlags::B
                    | ash::vk::ColorComponentFlags::A,
            )
            .blend_enable(true)
            .alpha_blend_op(ash::vk::BlendOp::ADD)
            .src_alpha_blend_factor(ash::vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(ash::vk::BlendFactor::ZERO)
            .color_blend_op(ash::vk::BlendOp::ADD)
            .src_color_blend_factor(ash::vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(ash::vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .build()];

        let colour_blend_state_info = ash::vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&attachment_states);

        let pipeline_layout_info = ash::vk::PipelineLayoutCreateInfo::builder()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&[]);

        let pipeline_layout = unsafe {
            device
                .handle()
                .create_pipeline_layout(&pipeline_layout_info, None)?
        };

        let depth_stencil_state_info = ash::vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(ash::vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let mut pipeline_rendering_info = ash::vk::PipelineRenderingCreateInfoKHR::builder()
            .color_attachment_formats(builder.colour_formats)
            .depth_attachment_format(builder.depth_format)
            .stencil_attachment_format(builder.stencil_format);

        let pipeline_info = ash::vk::GraphicsPipelineCreateInfo::builder()
            .push_next(&mut pipeline_rendering_info)
            .stages(&shader_stage_infos)
            .vertex_input_state(&input_state_info)
            .input_assembly_state(&input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterisation_state_info)
            .multisample_state(&multisample_state_info)
            .color_blend_state(&colour_blend_state_info)
            .layout(pipeline_layout)
            .dynamic_state(&dynamic_state_info)
            .depth_stencil_state(&depth_stencil_state_info)
            .build();

        let pipeline_results = unsafe {
            device.handle().create_graphics_pipelines(
                ash::vk::PipelineCache::null(),
                &[pipeline_info],
                None,
            )
        };

        // TODO: Figure out how to better handle this error.
        if pipeline_results.is_err() {
            return Err("The pipeline could not be created.".to_string().into());
        }

        let pipeline = pipeline_results.unwrap()[0];

        Ok(Self {
            pipeline_handle: pipeline,
            pipeline_layout,
        })
    }

    pub fn handle(&self) -> &ash::vk::Pipeline {
        &self.pipeline_handle
    }

    pub fn layout(&self) -> &ash::vk::PipelineLayout {
        &self.pipeline_layout
    }
}

impl Destroy for Pipeline {
    fn destroy(&mut self, device: &Device) {
        unsafe {
            device
                .handle()
                .destroy_pipeline_layout(self.pipeline_layout, None);
            device.handle().destroy_pipeline(self.pipeline_handle, None);
        }
    }
}
