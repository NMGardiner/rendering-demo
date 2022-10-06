#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texture_coords;
layout(location = 3) in ivec4 in_joint_indices;
layout(location = 4) in vec4 in_joint_weights;

layout(location = 0) out vec2 out_texture_coords;

layout(push_constant) uniform constants {
    mat4 matrix;
} push_constants;

layout(set = 1, binding = 1) readonly buffer JointBuffer {
    mat4 in_joint_matrices[];
};

void main() {
    mat4 skin_matrix = 
		in_joint_weights.x * in_joint_matrices[in_joint_indices.x] +
		in_joint_weights.y * in_joint_matrices[in_joint_indices.y] +
		in_joint_weights.z * in_joint_matrices[in_joint_indices.z] +
		in_joint_weights.w * in_joint_matrices[in_joint_indices.w];

    gl_Position = push_constants.matrix * skin_matrix * vec4(in_position, 1.0);
    out_texture_coords = in_texture_coords;
}