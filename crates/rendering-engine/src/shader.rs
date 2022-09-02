use std::ffi::CStr;
use std::{error::Error, fs::File, path::Path};

use crate::*;

use ash::vk::DescriptorType as AshDescriptorType;
use ash::vk::Format as AshFormat;
use ash::vk::ShaderStageFlags as AshShaderStageFlags;
use spirv_reflect::types::*;

/// A wrapper for a shader, storing information generated from shader reflection.
pub struct Shader {
    module: ash::vk::ShaderModule,
    shader_stage_info: ash::vk::PipelineShaderStageCreateInfo,
    binding_descriptions: Vec<ash::vk::VertexInputBindingDescription>,
    attribute_descriptions: Vec<ash::vk::VertexInputAttributeDescription>,
    push_constant_ranges: Vec<ash::vk::PushConstantRange>,
    descriptor_set_layouts: Vec<ash::vk::DescriptorSetLayout>,
}

impl Shader {
    /// Load a SPIR-V shader from a file, and use reflection to determine various pipeline layout settings.
    ///
    /// # Errors
    ///
    /// This function can error if opening the file fails, if `ash` fails to parse the SPIR-V or create the shader
    /// module or descriptor set layout, or if `spirv_reflect` fails to create a reflection module.
    pub fn new(path: &Path, device: &Device) -> Result<Self, Box<dyn Error>> {
        let mut file = File::open(path)?;
        let bytecode = ash::util::read_spv(&mut file)?;

        let module_info = ash::vk::ShaderModuleCreateInfo::builder().code(&bytecode);

        let module = unsafe { device.create_shader_module(&module_info, None)? };

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
                // TODO: Ignore any inputs with a location greater than maxVertexInputAttributes.
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

        let push_constant_ranges = reflection_module
            .enumerate_push_constant_blocks(None)?
            .iter()
            .map(|range| {
                ash::vk::PushConstantRange::builder()
                    .offset(range.offset)
                    .size(range.size)
                    .stage_flags(stage_flags)
                    .build()
            })
            .collect::<Vec<_>>();

        let mut descriptor_set_layouts: Vec<ash::vk::DescriptorSetLayout> = vec![];

        for descriptor_set in reflection_module.enumerate_descriptor_sets(None)? {
            let mut set_bindings: Vec<ash::vk::DescriptorSetLayoutBinding> = vec![];
            let mut set_binding_flags: Vec<ash::vk::DescriptorBindingFlags> = vec![];

            for set_binding in descriptor_set.bindings {
                // If the array length isn't specified, give it a default, non-zero value.
                // This is the situation when using bindless textures.
                let (descriptor_count, binding_flags) = if set_binding.array.dims.is_empty() {
                    (1024, ash::vk::DescriptorBindingFlags::PARTIALLY_BOUND)
                } else {
                    (set_binding.count, ash::vk::DescriptorBindingFlags::empty())
                };

                set_bindings.push(
                    ash::vk::DescriptorSetLayoutBinding::builder()
                        .descriptor_type(convert_descriptor_type(set_binding.descriptor_type))
                        .descriptor_count(descriptor_count)
                        .binding(set_binding.binding)
                        .stage_flags(stage_flags)
                        .build(),
                );

                set_binding_flags.push(binding_flags);
            }

            let mut set_layout_binding_flags =
                ash::vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                    .binding_flags(&set_binding_flags);

            let set_layout_info = ash::vk::DescriptorSetLayoutCreateInfo::builder()
                .push_next(&mut set_layout_binding_flags)
                .flags(ash::vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .bindings(set_bindings.as_slice());

            let set_layout =
                unsafe { device.create_descriptor_set_layout(&set_layout_info, None)? };

            descriptor_set_layouts.push(set_layout);
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
            push_constant_ranges,
            descriptor_set_layouts,
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

    pub fn push_constant_ranges(&self) -> &Vec<ash::vk::PushConstantRange> {
        &self.push_constant_ranges
    }

    pub fn descriptor_set_layouts(&self) -> &Vec<ash::vk::DescriptorSetLayout> {
        &self.descriptor_set_layouts
    }
}

impl Destroy for Shader {
    fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.module, None);
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

/// Convert spirv-reflect's descriptor type flags to those used by Ash.
fn convert_descriptor_type(reflect_descriptor_type: ReflectDescriptorType) -> AshDescriptorType {
    match reflect_descriptor_type {
        ReflectDescriptorType::AccelerationStructureNV => {
            AshDescriptorType::ACCELERATION_STRUCTURE_NV
        }
        ReflectDescriptorType::CombinedImageSampler => AshDescriptorType::COMBINED_IMAGE_SAMPLER,
        ReflectDescriptorType::InputAttachment => AshDescriptorType::INPUT_ATTACHMENT,
        ReflectDescriptorType::SampledImage => AshDescriptorType::SAMPLED_IMAGE,
        ReflectDescriptorType::Sampler => AshDescriptorType::SAMPLER,
        ReflectDescriptorType::StorageBuffer => AshDescriptorType::STORAGE_BUFFER,
        ReflectDescriptorType::StorageBufferDynamic => AshDescriptorType::UNIFORM_BUFFER_DYNAMIC,
        ReflectDescriptorType::StorageImage => AshDescriptorType::STORAGE_IMAGE,
        ReflectDescriptorType::StorageTexelBuffer => AshDescriptorType::STORAGE_TEXEL_BUFFER,
        ReflectDescriptorType::Undefined => AshDescriptorType::default(),
        ReflectDescriptorType::UniformBuffer => AshDescriptorType::UNIFORM_BUFFER,
        ReflectDescriptorType::UniformBufferDynamic => AshDescriptorType::UNIFORM_BUFFER_DYNAMIC,
        ReflectDescriptorType::UniformTexelBuffer => AshDescriptorType::UNIFORM_TEXEL_BUFFER,
    }
}
