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

/// An object representing a glTF scene, with the associated vertex/index buffers, textures, and materials.
pub struct Scene {
    nodes: Vec<Node>,
    root_node_indices: Vec<usize>,

    skins: Vec<Skin>,
    animations: Vec<Animation>,
    joint_buffer: Buffer,

    pub vertex_buffer: Buffer,
    pub index_buffer: Option<Buffer>,
    pub materials: Vec<MaterialData>,
}

impl Scene {
    // TODO: Does the below comment still apply?
    /// Load a given glTF file, including the vertex/index buffers, textures, and materials.
    /// Note: this currently only supports glTF files with a single scene.
    pub fn load(
        device: &Device,
        path: &str,
        textures: &mut slab::Slab<Image>,
    ) -> Result<Scene, Box<dyn Error>> {
        let (document, buffers, images) = gltf::import(Path::new(path))?;

        let mut scene_nodes: Vec<Node> = vec![];
        scene_nodes.resize_with(document.nodes().len(), Node::default);
        let mut scene_vertices: Vec<Vertex> = vec![];
        let mut scene_indices: Vec<u32> = vec![];
        let mut scene_materials: Vec<MaterialData> = vec![];

        for material in document.materials() {
            let mut material_data = MaterialData::default();

            let matrough_data = material.pbr_metallic_roughness();

            material_data.base_colour_factor = glm::make_vec4(&matrough_data.base_color_factor());

            // Albedo / base colour
            if let Some(albedo_texture_info) = matrough_data.base_color_texture() {
                let image = &images[albedo_texture_info.texture().index()];
                material_data.base_colour_texture =
                    textures.insert(texture_from_image(image, device)?) as i32;
            }

            // Metallic roughness
            if let Some(matrough_texture_info) = matrough_data.metallic_roughness_texture() {
                let image = &images[matrough_texture_info.texture().index()];
                material_data.matrough_texture =
                    textures.insert(texture_from_image(image, device)?) as i32;
            }

            // Normal
            if let Some(normal_texture_info) = material.normal_texture() {
                let image = &images[normal_texture_info.texture().index()];
                material_data.normal_texture =
                    textures.insert(texture_from_image(image, device)?) as i32;
            }

            // Ambient occlusion
            if let Some(occlusion_texture_info) = material.occlusion_texture() {
                let image = &images[occlusion_texture_info.texture().index()];
                material_data.occlusion_texture =
                    textures.insert(texture_from_image(image, device)?) as i32;
            }

            // Emissive
            if let Some(emissive_texture_info) = material.emissive_texture() {
                let image = &images[emissive_texture_info.texture().index()];
                material_data.emissive_texture =
                    textures.insert(texture_from_image(image, device)?) as i32;
            }

            scene_materials.push(material_data);
        }

        // Ensure there is a default material if none are included.
        if scene_materials.is_empty() {
            scene_materials.push(MaterialData::default());
        }

        let skins = document
            .skins()
            .map(|skin| {
                let reader = skin.reader(|buffer| Some(&buffers[buffer.index()]));

                let inverse_bind_matrices =
                    if let Some(inverse_bind_matrices) = reader.read_inverse_bind_matrices() {
                        inverse_bind_matrices
                            .map(|matrix| unsafe {
                                glm::make_mat4(std::slice::from_raw_parts(
                                    matrix.as_ptr().cast(),
                                    matrix.len() * 4,
                                ))
                            })
                            .collect()
                    } else {
                        vec![]
                    };

                let root_node_index = skin.skeleton().map(|skeleton| skeleton.index());

                let joint_node_indices = skin.joints().map(|joint| joint.index()).collect();

                Skin {
                    inverse_bind_matrices,
                    root_node_index,
                    joint_node_indices,
                }
            })
            .collect::<Vec<_>>();

        let animations = document
            .animations()
            .map(|animation| {
                let mut animation_start_time = std::f32::MAX;
                let mut animation_end_time = std::f32::MIN;

                let channels = animation
                    .channels()
                    .map(|channel| {
                        let reader = channel.reader(|buffer| Some(&buffers[buffer.index()]));

                        let inputs = if let Some(inputs) = reader.read_inputs() {
                            inputs
                                .map(|input| {
                                    animation_start_time = animation_start_time.min(input);
                                    animation_end_time = animation_end_time.max(input);

                                    input
                                })
                                .collect::<Vec<_>>()
                        } else {
                            vec![]
                        };

                        let (outputs, output_type) = if let Some(outputs) = reader.read_outputs() {
                            match outputs {
                                gltf::animation::util::ReadOutputs::Translations(translations) => (
                                    translations
                                        .map(|translation| {
                                            glm::vec4(
                                                translation[0],
                                                translation[1],
                                                translation[2],
                                                0.0,
                                            )
                                        })
                                        .collect(),
                                    AnimationType::Translation,
                                ),
                                gltf::animation::util::ReadOutputs::Rotations(rotations) => (
                                    rotations
                                        .into_f32()
                                        .map(|rotation| {
                                            glm::vec4(
                                                rotation[0],
                                                rotation[1],
                                                rotation[2],
                                                rotation[3],
                                            )
                                        })
                                        .collect(),
                                    AnimationType::Rotation,
                                ),
                                gltf::animation::util::ReadOutputs::Scales(scales) => (
                                    scales
                                        .map(|scale| glm::vec4(scale[0], scale[1], scale[2], 0.0))
                                        .collect(),
                                    AnimationType::Scale,
                                ),
                                gltf::animation::util::ReadOutputs::MorphTargetWeights(weights) => {
                                    (
                                        weights
                                            .into_f32()
                                            .map(|weight| glm::vec4(weight, 0.0, 0.0, 0.0))
                                            .collect(),
                                        AnimationType::MorphTargetWeight,
                                    )
                                }
                            }
                        } else {
                            // Just a dummy type.
                            (vec![], AnimationType::Translation)
                        };

                        let interpolation_method = match channel.sampler().interpolation() {
                            gltf::animation::Interpolation::Linear => InterpolationMethod::Linear,
                            gltf::animation::Interpolation::Step => InterpolationMethod::Step,
                            gltf::animation::Interpolation::CubicSpline => {
                                InterpolationMethod::CubicSpline
                            }
                        };

                        AnimationChannel {
                            node_index: channel.target().node().index(),
                            inputs,
                            output_type,
                            outputs,
                            interpolation_method,
                        }
                    })
                    .collect::<Vec<_>>();

                Animation {
                    channels,
                    start: animation_start_time,
                    end: animation_end_time,
                    current_time: 0.0,
                }
            })
            .collect::<Vec<_>>();

        let mut buffer_size = 0;
        for skin in skins.iter() {
            buffer_size += skin.inverse_bind_matrices.len() * std::mem::size_of::<glm::Mat4>();
        }

        let joint_buffer = Buffer::new(
            device,
            buffer_size as u64,
            ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::TRANSFER_DST,
            gpu_allocator::MemoryLocation::CpuToGpu,
        )?;

        let mut root_node_indices: Vec<usize> = vec![];

        for scene in document.scenes() {
            for node in scene.nodes() {
                root_node_indices.push(node.index());

                // Load all root nodes.
                Scene::load_node(
                    &node,
                    None,
                    &buffers,
                    &mut scene_nodes,
                    &mut scene_vertices,
                    &mut scene_indices,
                );
            }
        }

        let vertex_buffer = Buffer::new_with_data(
            device,
            ash::vk::BufferUsageFlags::VERTEX_BUFFER,
            &scene_vertices,
        )?;

        let index_buffer = if scene_indices.is_empty() {
            None
        } else {
            Some(Buffer::new_with_data(
                device,
                ash::vk::BufferUsageFlags::INDEX_BUFFER,
                &scene_indices,
            )?)
        };

        Ok(Self {
            nodes: scene_nodes,
            root_node_indices,
            skins,
            animations,
            joint_buffer,
            vertex_buffer,
            index_buffer,
            materials: scene_materials,
        })
    }

    fn load_node(
        gltf_node: &gltf::Node,
        parent_node_index: Option<usize>,
        buffers: &[gltf::buffer::Data],
        scene_nodes: &mut Vec<Node>,
        scene_vertices: &mut Vec<Vertex>,
        scene_indices: &mut Vec<u32>,
    ) {
        let (translation, rotation, scale) = gltf_node.transform().decomposed();

        let translation = glm::make_vec3(unsafe {
            std::slice::from_raw_parts(translation.as_ptr(), translation.len())
        });

        let rotation = glm::make_quat(unsafe {
            std::slice::from_raw_parts(rotation.as_ptr(), rotation.len())
        });

        let scale =
            glm::make_vec3(unsafe { std::slice::from_raw_parts(scale.as_ptr(), scale.len()) });

        for child_node in gltf_node.children() {
            Scene::load_node(
                &child_node,
                Some(gltf_node.index()),
                buffers,
                scene_nodes,
                scene_vertices,
                scene_indices,
            );
        }

        let mesh = if let Some(mesh) = gltf_node.mesh() {
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
                            .map(|coords| glm::make_vec2(coords.as_slice()))
                            .collect()
                    } else {
                        vec![]
                    };

                    let joint_indices = if let Some(joint_indices) = reader.read_joints(0) {
                        joint_indices
                            .into_u16()
                            .map(|indices| {
                                glm::make_vec4(indices.map(|value| value as u32).as_slice())
                            })
                            .collect()
                    } else {
                        vec![]
                    };

                    let joint_weights = if let Some(joint_weights) = reader.read_weights(0) {
                        joint_weights
                            .into_f32()
                            .map(|weights| glm::make_vec4(weights.as_slice()))
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
                            texture_coords: *tex_coords.get(index).unwrap_or(&glm::Vec2::zeros()),
                            joint_indices: *joint_indices
                                .get(index)
                                .unwrap_or(&glm::vec4(0, 0, 0, 0)),
                            joint_weights: *joint_weights.get(index).unwrap_or(&glm::Vec4::zeros()),
                        })
                        .collect();

                    let indices = if let Some(read_indices) = reader.read_indices() {
                        read_indices
                            .into_u32()
                            .map(|index| index + scene_vertices.len() as u32)
                            .collect()
                    } else {
                        vec![]
                    };

                    let primitive = Primitive {
                        first_index: scene_indices.len(),
                        index_count: indices.len(),
                        vertex_count: vertices.len(),
                        material_index: primitive.material().index(),
                    };

                    scene_vertices.extend(vertices);
                    scene_indices.extend(indices);

                    primitive
                })
                .collect();

            Some(Mesh { primitives })
        } else {
            None
        };

        scene_nodes[gltf_node.index()] = Node {
            parent_index: parent_node_index,
            child_indices: gltf_node.children().map(|child| child.index()).collect(),
            translation,
            rotation,
            scale,
            mesh,
            skin_index: gltf_node.skin().map(|skin| skin.index()),
        };
    }

    pub fn update_animations(&mut self, delta: f64) {
        let mut animation = &mut self.animations[0];

        animation.current_time += delta;

        if animation.current_time > animation.end as f64 {
            animation.current_time -= animation.end as f64;
        }

        for channel in animation.channels.iter() {
            for (index, window) in channel.inputs.windows(2).enumerate() {
                let current_input = window[0];
                let next_input = window[1];

                if animation.current_time >= current_input.into()
                    && animation.current_time <= next_input.into()
                {
                    let interpolation = (animation.current_time - current_input as f64)
                        / (next_input as f64 - current_input as f64);

                    let node = &mut self.nodes[channel.node_index];

                    match channel.interpolation_method {
                        InterpolationMethod::Linear => match channel.output_type {
                            AnimationType::Translation => {
                                node.translation = glm::mix(
                                    &channel.outputs[index].xyz(),
                                    &channel.outputs[index + 1].xyz(),
                                    interpolation as f32,
                                );
                            }
                            AnimationType::Rotation => {
                                node.rotation = glm::quat_normalize(&glm::quat_slerp(
                                    &glm::make_quat(channel.outputs[index].as_slice()),
                                    &glm::make_quat(channel.outputs[index + 1].as_slice()),
                                    interpolation as f32,
                                ));
                            }
                            AnimationType::Scale => {
                                node.scale = glm::mix(
                                    &channel.outputs[index].xyz(),
                                    &channel.outputs[index + 1].xyz(),
                                    interpolation as f32,
                                );
                            }
                            AnimationType::MorphTargetWeight => {
                                log::warn!("Morph target weights are unsupported!");
                            }
                        },
                        InterpolationMethod::Step => {
                            log::warn!("Step interpolation is unsupported!");
                        }
                        InterpolationMethod::CubicSpline => {
                            log::warn!("CubicSpline interpolation is unsupported!");
                        }
                    }
                }
            }
        }

        self.update_joints();
    }

    pub fn update_joints(&mut self) {
        for node_index in self.root_node_indices.clone() {
            self.update_joint(node_index);
        }
    }

    pub fn update_joint(&mut self, node_index: usize) {
        let node = &self.nodes[node_index];

        if let Some(skin_index) = node.skin_index {
            let skin = &self.skins[skin_index];

            let inverse_transform = glm::inverse(&self.get_node_matrix(node_index));

            let mut joint_matrices = vec![glm::Mat4::identity(); skin.joint_node_indices.len()];

            for (index, joint_node_index) in skin.joint_node_indices.iter().enumerate() {
                let joint_node_matrix = self.get_node_matrix(*joint_node_index);
                let joint_inverse_bind_matrix = skin.inverse_bind_matrices[index];
                joint_matrices[index] = joint_node_matrix * joint_inverse_bind_matrix;
                joint_matrices[index] = inverse_transform * joint_matrices[index];
            }

            unsafe {
                let memory_pointer = self
                    .joint_buffer
                    .allocation()
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr() as *mut glm::Mat4;

                memory_pointer
                    .copy_from_nonoverlapping(joint_matrices.as_ptr(), joint_matrices.len());
            }
        }

        for child_index in node.child_indices.clone() {
            self.update_joint(child_index);
        }
    }

    pub fn get_node_matrix(&self, node_index: usize) -> glm::Mat4 {
        let node = &self.nodes[node_index];

        let mut matrix = node.get_local_matrix();
        let mut parent_index: Option<usize> = node.parent_index;
        while parent_index.is_some() {
            matrix = self.nodes[parent_index.unwrap()].get_local_matrix() * matrix;

            parent_index = self.nodes[parent_index.unwrap()].parent_index;
        }

        matrix
    }

    /// Draw all nodes in the scene.
    pub fn draw(
        &self,
        device: &Device,
        command_buffer: &ash::vk::CommandBuffer,
        pipeline_layout: &ash::vk::PipelineLayout,
        pv_matrix: glm::Mat4,
    ) {
        // Draw the root nodes.
        for root_node_index in self.root_node_indices.iter() {
            self.draw_node(
                *root_node_index,
                device,
                command_buffer,
                pipeline_layout,
                pv_matrix,
            );
        }
    }

    fn draw_node(
        &self,
        node_index: usize,
        device: &Device,
        command_buffer: &ash::vk::CommandBuffer,
        pipeline_layout: &ash::vk::PipelineLayout,
        pv_matrix: glm::Mat4,
    ) {
        let node = &self.nodes[node_index];

        if let Some(mesh) = &node.mesh {
            // Handle the transform matrix & push constants.
            let push_constants = crate::PushConstants {
                matrix: pv_matrix * self.get_node_matrix(node_index),
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
                } else if primitive.vertex_count > 0 {
                    unsafe {
                        device.cmd_draw(*command_buffer, primitive.vertex_count as u32, 1, 0, 0);
                    }
                }
            }
        }

        for child_node_index in node.child_indices.iter() {
            self.draw_node(
                *child_node_index,
                device,
                command_buffer,
                pipeline_layout,
                pv_matrix,
            );
        }
    }

    pub fn get_joint_buffer(&self) -> &Buffer {
        &self.joint_buffer
    }

    pub fn set_scale(&mut self, scale: f32) {
        for root_node_index in self.root_node_indices.iter() {
            self.nodes[*root_node_index].scale *= scale;
        }
    }
}

impl Destroy for Scene {
    fn destroy(&mut self, device: &Device) {
        self.joint_buffer.destroy(device);

        self.vertex_buffer.destroy(device);

        if let Some(index_buffer) = &mut self.index_buffer {
            index_buffer.destroy(device);
        }
    }
}

struct Skin {
    inverse_bind_matrices: Vec<glm::Mat4>,
    root_node_index: Option<usize>,
    joint_node_indices: Vec<usize>,
}

#[derive(Default)]
struct Node {
    parent_index: Option<usize>,
    child_indices: Vec<usize>,

    translation: glm::Vec3,
    rotation: glm::Quat,
    scale: glm::Vec3,

    mesh: Option<Mesh>,

    skin_index: Option<usize>,
}

impl Node {
    fn get_local_matrix(&self) -> glm::Mat4 {
        glm::translate(&glm::Mat4::identity(), &self.translation)
            * glm::quat_cast(&self.rotation)
            * glm::scale(&glm::Mat4::identity(), &self.scale)
    }
}

struct Mesh {
    primitives: Vec<Primitive>,
}

struct Primitive {
    first_index: usize,
    index_count: usize,
    vertex_count: usize,
    material_index: Option<usize>,
}

struct Animation {
    channels: Vec<AnimationChannel>,
    start: f32,
    end: f32,
    current_time: f64,
}

enum AnimationType {
    Translation,
    Rotation,
    Scale,
    MorphTargetWeight,
}

enum InterpolationMethod {
    Linear,
    Step,
    CubicSpline,
}

struct AnimationChannel {
    node_index: usize,
    inputs: Vec<f32>,
    output_type: AnimationType,
    outputs: Vec<glm::Vec4>,
    interpolation_method: InterpolationMethod,
}
