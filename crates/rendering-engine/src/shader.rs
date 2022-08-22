use std::ffi::CStr;
use std::{error::Error, fs::File, path::Path};

use crate::*;

use ash::vk::Format as AshFormat;
use ash::vk::ShaderStageFlags as AshShaderStageFlags;
use spirv_reflect::types::*;

/// A wrapper for a shader, storing information generated from shader reflection.
pub struct Shader {
    module: ash::vk::ShaderModule,
    shader_stage_info: ash::vk::PipelineShaderStageCreateInfo,
    binding_descriptions: Vec<ash::vk::VertexInputBindingDescription>,
    attribute_descriptions: Vec<ash::vk::VertexInputAttributeDescription>,
}

impl Shader {
    /// Load a SPIR-V shader from a file,
    pub fn new(path: &Path, device: &Device) -> Result<Self, Box<dyn Error>> {
        let mut file = File::open(path)?;
        let bytecode = ash::util::read_spv(&mut file)?;

        let module_info = ash::vk::ShaderModuleCreateInfo::builder().code(&bytecode);

        let module = unsafe { device.handle().create_shader_module(&module_info, None)? };

        let reflection_module = spirv_reflect::ShaderModule::load_u32_data(&bytecode)?;

        let stage_flags = convert_shader_stage_flags(reflection_module.get_shader_stage());

        let mut binding_descriptions: Vec<ash::vk::VertexInputBindingDescription> = Vec::new();
        let mut attribute_descriptions: Vec<ash::vk::VertexInputAttributeDescription> = Vec::new();

        if stage_flags.contains(ash::vk::ShaderStageFlags::VERTEX) {
            let mut binding_description = ash::vk::VertexInputBindingDescription::builder()
                .binding(0)
                .stride(0)
                .input_rate(ash::vk::VertexInputRate::VERTEX)
                .build();

            for input in reflection_module.enumerate_input_variables(None)? {
                // gl_VertexIndex seems to be passed in at this location, so ignore it.
                if input.location == std::u32::MAX {
                    continue;
                }

                let (format, format_size) = convert_format(input.format);
                attribute_descriptions.push(
                    ash::vk::VertexInputAttributeDescription::builder()
                        .binding(0)
                        .location(input.location)
                        .format(format)
                        // Store the size of the format for the final offset calculation later.
                        .offset(format_size)
                        .build(),
                );
            }

            // If there's no attribute descriptions, then there's no need for the binding description.
            if !attribute_descriptions.is_empty() {
                // Sort by location, and calculate the correct offsets.
                attribute_descriptions.sort_by(|a, b| a.location.partial_cmp(&b.location).unwrap());

                for attribute_description in attribute_descriptions.iter_mut() {
                    let format_size = attribute_description.offset;
                    attribute_description.offset = binding_description.stride;
                    binding_description.stride += format_size;
                }

                binding_descriptions.push(binding_description);
            }
        }

        let shader_stage_info = ash::vk::PipelineShaderStageCreateInfo::builder()
            .module(module)
            .stage(stage_flags)
            // The string needs to be static to avoid going out of scope.
            .name(CStr::from_bytes_with_nul(b"main\0")?)
            .build();

        Ok(Self {
            module,
            shader_stage_info,
            binding_descriptions,
            attribute_descriptions,
        })
    }

    pub fn stage_info(&self) -> &ash::vk::PipelineShaderStageCreateInfo {
        &self.shader_stage_info
    }

    pub fn binding_descriptions(&self) -> &Vec<ash::vk::VertexInputBindingDescription> {
        &self.binding_descriptions
    }

    pub fn attribute_descriptions(&self) -> &Vec<ash::vk::VertexInputAttributeDescription> {
        &self.attribute_descriptions
    }
}

impl Destroy for Shader {
    fn destroy(&mut self, device: &Device) {
        unsafe {
            device.handle().destroy_shader_module(self.module, None);
        }
    }
}

/// Convert spirv-reflect's stage flags to those used by Ash.
fn convert_shader_stage_flags(reflect_stage: ReflectShaderStageFlags) -> AshShaderStageFlags {
    let mut shader_stage_flags = AshShaderStageFlags::empty();

    if reflect_stage.contains(ReflectShaderStageFlags::ANY_HIT_BIT_NV) {
        shader_stage_flags |= AshShaderStageFlags::ANY_HIT_NV;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::CALLABLE_BIT_NV) {
        shader_stage_flags |= AshShaderStageFlags::CALLABLE_NV;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::CLOSEST_HIT_BIT_NV) {
        shader_stage_flags |= AshShaderStageFlags::CLOSEST_HIT_NV;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::COMPUTE) {
        shader_stage_flags |= AshShaderStageFlags::COMPUTE;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::FRAGMENT) {
        shader_stage_flags |= AshShaderStageFlags::FRAGMENT;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::GEOMETRY) {
        shader_stage_flags |= AshShaderStageFlags::GEOMETRY;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::INTERSECTION_BIT_NV) {
        shader_stage_flags |= AshShaderStageFlags::INTERSECTION_NV;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::MISS_BIT_NV) {
        shader_stage_flags |= AshShaderStageFlags::MISS_NV;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::RAYGEN_BIT_NV) {
        shader_stage_flags |= AshShaderStageFlags::RAYGEN_NV;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::TESSELLATION_CONTROL) {
        shader_stage_flags |= AshShaderStageFlags::TESSELLATION_CONTROL;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::TESSELLATION_EVALUATION) {
        shader_stage_flags |= AshShaderStageFlags::TESSELLATION_EVALUATION;
    }
    if reflect_stage.contains(ReflectShaderStageFlags::UNDEFINED) {
        shader_stage_flags |= AshShaderStageFlags::empty();
    }
    if reflect_stage.contains(ReflectShaderStageFlags::VERTEX) {
        shader_stage_flags |= AshShaderStageFlags::VERTEX;
    }

    shader_stage_flags
}

/// Convert spirv-reflect's format flags to those used by Ash.
fn convert_format(reflect_format: ReflectFormat) -> (ash::vk::Format, u32) {
    match reflect_format {
        ReflectFormat::R32G32B32A32_SFLOAT => (AshFormat::R32G32B32A32_SFLOAT, 16),
        ReflectFormat::R32G32B32A32_SINT => (AshFormat::R32G32B32A32_SINT, 16),
        ReflectFormat::R32G32B32A32_UINT => (AshFormat::R32G32B32A32_UINT, 16),
        ReflectFormat::R32G32B32_SFLOAT => (AshFormat::R32G32B32_SFLOAT, 12),
        ReflectFormat::R32G32B32_SINT => (AshFormat::R32G32B32_SINT, 12),
        ReflectFormat::R32G32B32_UINT => (AshFormat::R32G32B32_UINT, 12),
        ReflectFormat::R32G32_SFLOAT => (AshFormat::R32G32_SFLOAT, 8),
        ReflectFormat::R32G32_SINT => (AshFormat::R32G32_SINT, 8),
        ReflectFormat::R32G32_UINT => (AshFormat::R32G32_UINT, 8),
        ReflectFormat::R32_SFLOAT => (AshFormat::R32_SFLOAT, 4),
        ReflectFormat::R32_SINT => (AshFormat::R32_SINT, 4),
        ReflectFormat::R32_UINT => (AshFormat::R32_UINT, 4),
        ReflectFormat::Undefined => (AshFormat::UNDEFINED, 0),
    }
}