#version 450

#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec3 in_colour;
layout(location = 1) in vec2 in_texture_coords;
layout(location = 2) flat in int in_texture_index;

layout(location = 0) out vec4 out_colour;

layout(set = 0, binding = 0) uniform sampler2D in_textures[];

void main() {
    out_colour = texture(in_textures[in_texture_index], in_texture_coords);
}