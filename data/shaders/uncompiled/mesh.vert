#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_texture_coords;

layout(location = 0) out vec2 out_texture_coords;

layout(push_constant) uniform constants {
    mat4 matrix;
} push_constants;

void main() {
    gl_Position = push_constants.matrix * vec4(in_position, 1.0);
    out_texture_coords = in_texture_coords;
}