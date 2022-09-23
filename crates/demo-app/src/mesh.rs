use std::{error::Error, path::Path};

use rendering_engine::*;

use nalgebra_glm as glm;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: glm::Vec3,
    pub normal: glm::Vec3,
    pub texture_coords: glm::Vec2,
    pub joint_indices: glm::IVec4,
    pub joint_weights: glm::Vec4,
}

pub struct PushConstants {
    pub matrix: glm::Mat4,
}

/// Object for storing per-material data including indexes into the texture array.
pub struct MaterialData {
    pub base_colour_factor: glm::Vec4,

    pub base_colour_texture: i32,
    pub matrough_texture: i32,
    pub normal_texture: i32,
    pub occlusion_texture: i32,

    pub emissive_texture: i32,
    pub padding0: u32,
    pub padding1: u32,
    pub padding2: u32,
}

impl Default for MaterialData {
    fn default() -> Self {
        Self {
            base_colour_factor: glm::vec4(1.0, 0.0, 0.0, 1.0),
            base_colour_texture: -1,
            matrough_texture: -1,
            normal_texture: -1,
            occlusion_texture: -1,
            emissive_texture: -1,
            padding0: 0,
            padding1: 0,
            padding2: 0,
        }
    }
}

pub trait Draw {
    fn draw(
        &self,
        device: &Device,
        command_buffer: &ash::vk::CommandBuffer,
        pipeline_layout: &ash::vk::PipelineLayout,
        pv_matrix: glm::Mat4,
    );
}

/// Offsets take the form (start, count).
pub struct Mesh {
    pub vertex_buffer: Buffer,
    pub vertex_offsets: Vec<(usize, usize)>,
    // glTF models may not provide indices.
    pub index_buffer: Option<Buffer>,
    pub index_offsets: Vec<(usize, usize)>,
    pub matrices: Vec<Option<glm::Mat4>>,
    pub textures: Vec<Image>,
    pub materials: Vec<MaterialData>,
    pub model_transform: glm::Mat4,
}

impl Mesh {
    /// Load a mesh from a .gltf/.glb file at the given path, creating the necessary textures.
    pub fn new(
        device: &Device,
        path: &str,
        position: glm::Vec3,
        scale: f32,
    ) -> Result<Self, Box<dyn Error>> {
        let (document, buffers, images) = gltf::import(Path::new(path))?;

        let mut vertices: Vec<Vertex> = vec![];
        let mut indices: Vec<u32> = vec![];
        let mut vertex_offsets: Vec<(usize, usize)> = vec![];
        let mut index_offsets: Vec<(usize, usize)> = vec![];
        let mut matrices: Vec<Option<glm::Mat4>> = vec![];
        let mut textures: Vec<Image> = vec![];
        let mut materials: Vec<MaterialData> = vec![];

        for material in document.materials() {
            let mut material_data = MaterialData::default();

            let matrough_data = material.pbr_metallic_roughness();

            material_data.base_colour_factor = glm::make_vec4(&matrough_data.base_color_factor());

            // Albedo / base colour
            if let Some(albedo_texture_info) = matrough_data.base_color_texture() {
                let image = &images[albedo_texture_info.texture().index()];
                material_data.base_colour_texture = textures.len() as i32;
                textures.push(texture_from_image(image, device)?);
            }

            // Metallic roughness
            if let Some(matrough_texture_info) = matrough_data.metallic_roughness_texture() {
                let image = &images[matrough_texture_info.texture().index()];
                material_data.matrough_texture = textures.len() as i32;
                textures.push(texture_from_image(image, device)?);
            }

            // Normal
            if let Some(normal_texture_info) = material.normal_texture() {
                let image = &images[normal_texture_info.texture().index()];
                material_data.normal_texture = textures.len() as i32;
                textures.push(texture_from_image(image, device)?);
            }

            // Ambient occlusion
            if let Some(occlusion_texture_info) = material.occlusion_texture() {
                let image = &images[occlusion_texture_info.texture().index()];
                material_data.occlusion_texture = textures.len() as i32;
                textures.push(texture_from_image(image, device)?);
            }

            // Emissive
            if let Some(emissive_texture_info) = material.emissive_texture() {
                let image = &images[emissive_texture_info.texture().index()];
                material_data.emissive_texture = textures.len() as i32;
                textures.push(texture_from_image(image, device)?);
            }

            materials.push(material_data);
        }

        // Ensure there is a default material if none are included.
        if materials.is_empty() {
            materials.push(MaterialData::default());
        }

        for node in document.nodes() {
            let transform_matrix = node.transform().matrix();
            let transform_matrix: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    transform_matrix.as_ptr().cast(),
                    transform_matrix.len() * 4,
                )
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
                    let mut joint_indices: Vec<glm::IVec4> = vec![];
                    let mut joint_weights: Vec<glm::Vec4> = vec![];

                    if let Some(primitive_positions) = reader.read_positions() {
                        for primitive_position in primitive_positions {
                            positions.push(glm::make_vec3(&primitive_position));

                            // Push a default normal in case there's none given.
                            normals.push(glm::Vec3::zeros());

                            texture_coords.push(glm::Vec2::zeros());

                            joint_indices.push(glm::vec4(-1, -1, -1, -1));

                            joint_weights.push(glm::vec4(0.0, 0.0, 0.0, 0.0));
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

                    if let Some(primitive_joint_indices) = reader.read_joints(0) {
                        match primitive_joint_indices {
                            gltf::mesh::util::ReadJoints::U8(data) => {
                                for (index, joint_data) in data.enumerate() {
                                    joint_indices[index] = glm::vec4(
                                        joint_data[0] as i32,
                                        joint_data[1] as i32,
                                        joint_data[2] as i32,
                                        joint_data[3] as i32,
                                    );
                                }
                            }
                            gltf::mesh::util::ReadJoints::U16(data) => {
                                for (index, joint_data) in data.enumerate() {
                                    joint_indices[index] = glm::vec4(
                                        joint_data[0] as i32,
                                        joint_data[1] as i32,
                                        joint_data[2] as i32,
                                        joint_data[3] as i32,
                                    );
                                }
                            }
                        }
                    }

                    if let Some(primitive_joint_weights) = reader.read_weights(0) {
                        match primitive_joint_weights {
                            gltf::mesh::util::ReadWeights::U8(data) => {
                                for (index, joint_data) in data.enumerate() {
                                    joint_weights[index] = glm::vec4(
                                        joint_data[0] as f32,
                                        joint_data[1] as f32,
                                        joint_data[2] as f32,
                                        joint_data[3] as f32,
                                    );
                                }
                            }
                            gltf::mesh::util::ReadWeights::U16(data) => {
                                for (index, joint_data) in data.enumerate() {
                                    joint_weights[index] = glm::vec4(
                                        joint_data[0] as f32,
                                        joint_data[1] as f32,
                                        joint_data[2] as f32,
                                        joint_data[3] as f32,
                                    );
                                }
                            }
                            gltf::mesh::util::ReadWeights::F32(data) => {
                                for (index, joint_data) in data.enumerate() {
                                    joint_weights[index] = glm::vec4(
                                        joint_data[0],
                                        joint_data[1],
                                        joint_data[2],
                                        joint_data[3],
                                    );
                                }
                            }
                        }
                    }

                    vertex_offsets.push((vertices.len(), positions.len()));
                    for (index, position) in positions.iter().enumerate() {
                        vertices.push(Vertex {
                            position: *position,
                            normal: normals[index],
                            texture_coords: texture_coords[index],
                            joint_indices: joint_indices[index],
                            joint_weights: joint_weights[index],
                        });
                    }

                    let start_index = indices.len();
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

                    index_offsets.push((start_index, indices.len() - start_index));
                }
            };
        }

        let vertex_buffer =
            Buffer::new_with_data(device, ash::vk::BufferUsageFlags::VERTEX_BUFFER, &vertices)?;

        let index_buffer = if indices.is_empty() {
            None
        } else {
            Some(Buffer::new_with_data(
                device,
                ash::vk::BufferUsageFlags::INDEX_BUFFER,
                &indices,
            )?)
        };

        // Translation * scale.
        let model_transform = glm::translate(&glm::Mat4::identity(), &position)
            * glm::scale(&glm::Mat4::identity(), &glm::vec3(scale, scale, scale));

        Ok(Self {
            vertex_buffer,
            vertex_offsets,
            index_buffer,
            index_offsets,
            matrices,
            textures,
            materials,
            model_transform,
        })
    }
}

impl Draw for Mesh {
    fn draw(
        &self,
        device: &Device,
        command_buffer: &ash::vk::CommandBuffer,
        pipeline_layout: &ash::vk::PipelineLayout,
        pv_matrix: glm::Mat4,
    ) {
        for (index, vertex_offset) in self.vertex_offsets.iter().enumerate() {
            if let Some(matrix) = self.matrices[index] {
                let push_constants = PushConstants {
                    matrix: pv_matrix * (self.model_transform * matrix),
                };

                let push_constants_slice = unsafe {
                    std::slice::from_raw_parts(
                        &push_constants as *const PushConstants as *const u8,
                        std::mem::size_of::<PushConstants>() / std::mem::size_of::<u8>(),
                    )
                };

                unsafe {
                    device.cmd_push_constants(
                        *command_buffer,
                        *pipeline_layout,
                        ash::vk::ShaderStageFlags::VERTEX,
                        0,
                        push_constants_slice,
                    );
                }
            }

            if self.index_buffer.is_some() {
                unsafe {
                    device.cmd_draw_indexed(
                        *command_buffer,
                        self.index_offsets[index].1 as u32,
                        1,
                        self.index_offsets[index].0 as u32,
                        vertex_offset.0 as i32,
                        0,
                    );
                }
            } else {
                unsafe {
                    device.cmd_draw(
                        *command_buffer,
                        vertex_offset.1 as u32,
                        1,
                        vertex_offset.0 as u32,
                        0,
                    );
                }
            }
        }
    }
}

impl Destroy for Mesh {
    fn destroy(&mut self, device: &Device) {
        for texture in self.textures.iter_mut() {
            texture.destroy(device);
        }

        self.vertex_buffer.destroy(device);

        if let Some(index_buffer) = &mut self.index_buffer {
            index_buffer.destroy(device);
        }
    }
}

fn texture_from_image(image: &gltf::image::Data, device: &Device) -> Result<Image, Box<dyn Error>> {
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

    Image::new_with_data(
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
    )
}
