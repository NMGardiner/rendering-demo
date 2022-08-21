#version 450

vec3 positions[3] = vec3[](
    vec3(0.0, 0.5, 1.0),
    vec3(0.5, -0.5, 1.0),
    vec3(-0.5, -0.5, 1.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 1.0);
}