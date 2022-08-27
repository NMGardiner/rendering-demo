#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_colour;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec2 in_texture_coords;

layout(location = 0) out vec3 out_colour;
layout(location = 1) out vec2 out_texture_coords;
layout(location = 2) out int out_texture_index;

layout(push_constant) uniform constants {
    mat4 matrix;
} push_constants;

void main() {
    gl_Position = push_constants.matrix * vec4(in_position, 1.0);
    out_colour = vec3(1.0, 1.0, 1.0) * dot(in_normal, vec3(0.58, 0.58, 0.58));
    out_texture_coords = in_texture_coords;
    out_texture_index =  gl_InstanceIndex;
}