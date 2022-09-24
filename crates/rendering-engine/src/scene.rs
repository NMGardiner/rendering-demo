use std::{error::Error, path::Path};

use nalgebra_glm as glm;

use crate::*;

#[derive(Copy, Clone)]
pub struct Vertex {
    pub position: glm::Vec3,
    pub normal: glm::Vec3,
    pub texture_coords: glm::Vec2,
    pub joint_indices: glm::UVec4,
    pub joint_weights: glm::Vec4,
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

pub struct Scene {
    root_node_indices: Vec<usize>,
    nodes: Vec<Node>,
}

impl Scene {
    pub fn load(device: &Device, path: &str) -> Result<Self, Box<dyn Error>> {
        let (document, buffers, images) = gltf::import(Path::new(path))?;

        let mut scene_vertices: Vec<Vertex> = vec![];
        let mut scene_indices: Vec<u32> = vec![];
        let mut scene_textures: Vec<Image> = vec![];
        let mut scene_materials: Vec<MaterialData> = vec![];

        for material in document.materials() {
            let mut material_data = MaterialData::default();

            let matrough_data = material.pbr_metallic_roughness();

            material_data.base_colour_factor = glm::make_vec4(&matrough_data.base_color_factor());

            // Albedo / base colour
            if let Some(albedo_texture_info) = matrough_data.base_color_texture() {
                let image = &images[albedo_texture_info.texture().index()];
                material_data.base_colour_texture = scene_textures.len() as i32;
                scene_textures.push(texture_from_image(image, device)?);
            }

            // Metallic roughness
            if let Some(matrough_texture_info) = matrough_data.metallic_roughness_texture() {
                let image = &images[matrough_texture_info.texture().index()];
                material_data.matrough_texture = scene_textures.len() as i32;
                scene_textures.push(texture_from_image(image, device)?);
            }

            // Normal
            if let Some(normal_texture_info) = material.normal_texture() {
                let image = &images[normal_texture_info.texture().index()];
                material_data.normal_texture = scene_textures.len() as i32;
                scene_textures.push(texture_from_image(image, device)?);
            }

            // Ambient occlusion
            if let Some(occlusion_texture_info) = material.occlusion_texture() {
                let image = &images[occlusion_texture_info.texture().index()];
                material_data.occlusion_texture = scene_textures.len() as i32;
                scene_textures.push(texture_from_image(image, device)?);
            }

            // Emissive
            if let Some(emissive_texture_info) = material.emissive_texture() {
                let image = &images[emissive_texture_info.texture().index()];
                material_data.emissive_texture = scene_textures.len() as i32;
                scene_textures.push(texture_from_image(image, device)?);
            }

            scene_materials.push(material_data);
        }

        // Ensure there is a default material if none are included.
        if scene_materials.is_empty() {
            scene_materials.push(MaterialData::default());
        }

        for node in document.nodes() {
            let transform_matrix = node.transform().matrix();
            let transform_matrix: &[f32] = unsafe {
                std::slice::from_raw_parts(
                    transform_matrix.as_ptr().cast(),
                    transform_matrix.len() * 4,
                )
            };

            for child in node.children() {
                child.index();
            }

            if let Some(mesh) = node.mesh() {
                let primitives: Vec<Primitive> = mesh
                    .primitives()
                    .map(|primitive| {
                        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                        let positions = if let Some(positions) = reader.read_positions() {
                            positions
                                .map(|position| glm::make_vec3(&position))
                                .collect()
                        } else {
                            vec![]
                        };

                        let normals = if let Some(normals) = reader.read_normals() {
                            normals.map(|normal| glm::make_vec3(&normal)).collect()
                        } else {
                            vec![]
                        };

                        let tex_coords = if let Some(tex_coords) = reader.read_tex_coords(0) {
                            tex_coords
                                .into_f32()
                                .map(|coords| glm::vec2(coords[0], coords[1]))
                                .collect()
                        } else {
                            vec![]
                        };

                        let joint_indices = if let Some(joint_indices) = reader.read_joints(0) {
                            joint_indices
                                .into_u16()
                                .map(|indices| {
                                    glm::vec4(
                                        indices[0] as u32,
                                        indices[1] as u32,
                                        indices[2] as u32,
                                        indices[3] as u32,
                                    )
                                })
                                .collect()
                        } else {
                            vec![]
                        };

                        let joint_weights = if let Some(joint_weights) = reader.read_weights(0) {
                            joint_weights
                                .into_f32()
                                .map(|weights| {
                                    glm::vec4(weights[0], weights[1], weights[2], weights[3])
                                })
                                .collect()
                        } else {
                            vec![]
                        };

                        let vertices: Vec<Vertex> = positions
                            .iter()
                            .enumerate()
                            .map(|(index, position)| Vertex {
                                position: *position,
                                normal: if normals.is_empty() {
                                    glm::Vec3::zeros()
                                } else {
                                    normals[index]
                                },
                                texture_coords: *tex_coords
                                    .get(index)
                                    .unwrap_or(&glm::Vec2::zeros()),
                                joint_indices: *joint_indices
                                    .get(index)
                                    .unwrap_or(&glm::vec4(0, 0, 0, 0)),
                                joint_weights: *joint_weights
                                    .get(index)
                                    .unwrap_or(&glm::Vec4::zeros()),
                            })
                            .collect();

                        let indices = if let Some(indices) = reader.read_indices() {
                            indices.into_u32().collect()
                        } else {
                            vec![]
                        };

                        let primitive = Primitive {
                            first_index: scene_indices.len(),
                            index_count: indices.len(),
                            material_index: primitive.material().index(),
                        };

                        scene_vertices.extend(vertices);
                        scene_indices.extend(indices);

                        primitive
                    })
                    .collect();
            }
        }

        Ok(Self {
            root_node_indices: vec![],
            nodes: vec![],
        })
    }

    pub fn draw(
        &self,
        device: &Device,
        command_buffer: &ash::vk::CommandBuffer,
        pipeline_layout: &ash::vk::PipelineLayout,
    ) {
        for root_node_index in self.root_node_indices.iter() {
            self.draw_node(*root_node_index, device, command_buffer, pipeline_layout);
        }
    }

    fn draw_node(
        &self,
        node_index: usize,
        device: &Device,
        command_buffer: &ash::vk::CommandBuffer,
        pipeline_layout: &ash::vk::PipelineLayout,
    ) {
        self.nodes[node_index].draw(self, device, command_buffer, pipeline_layout);
    }
}

struct Node {
    child_node_indices: Vec<usize>,
    matrix: Option<glm::Mat4>,
    mesh: Option<Mesh>,
}

impl Node {
    fn draw(
        &self,
        scene: &Scene,
        device: &Device,
        command_buffer: &ash::vk::CommandBuffer,
        pipeline_layout: &ash::vk::PipelineLayout,
    ) {
        if let Some(mesh) = &self.mesh {
            // Handle the transform matrix & push constants.

            for primitive in mesh.primitives.iter() {
                if primitive.index_count > 0 {
                    unsafe {
                        device.cmd_draw_indexed(
                            *command_buffer,
                            primitive.index_count as u32,
                            1,
                            primitive.first_index as u32,
                            0,
                            0,
                        );
                    }
                }
            }
        }

        for child_node_index in self.child_node_indices.iter() {
            scene.draw_node(*child_node_index, device, command_buffer, pipeline_layout);
        }
    }
}

struct Mesh {
    primitives: Vec<Primitive>,
}

impl Mesh {}

struct Primitive {
    first_index: usize,
    index_count: usize,
    material_index: Option<usize>,
}

impl Primitive {}
