#version 450

#extension GL_EXT_nonuniform_qualifier : enable

layout(location = 0) in vec2 in_texture_coords;

layout(location = 0) out vec4 out_colour;

struct MaterialData {
    vec4 base_colour_factor;

    int base_colour_texture;
    int matrough_texture;
    int normal_texture;
    int occlusion_texture;

    int emissive_texture;
    uint padding0;
    uint padding1;
    uint padding2;
};

layout(set = 0, binding = 0) readonly buffer MaterialBuffer {
    MaterialData materials[];
} in_materials;

layout(set = 1, binding = 0) uniform sampler2D in_textures[];

void main() {
    MaterialData material_data = in_materials.materials[0];

    // By default, use the base colour factor.
    out_colour = material_data.base_colour_factor;

    // If there's a base colour texture, use that instead of the base colour factor.
    if (material_data.base_colour_texture > -1) {
        vec4 base_colour = texture(in_textures[material_data.base_colour_texture], in_texture_coords);
        out_colour = base_colour;
    }
    
    // If there's an emissive texture, apply it.
    if (material_data.emissive_texture > -1) {
            vec4 emissive_colour = texture(in_textures[material_data.emissive_texture], in_texture_coords);
            out_colour += emissive_colour;
    }
}