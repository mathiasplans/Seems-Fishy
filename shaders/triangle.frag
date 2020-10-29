#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0) buffer Spheres {
    uint count;
    vec3 positions[];
};

layout(set = 1, binding = 0) buffer Squares {
    uint count;
    vec3 positions[];
};

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 pos;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(pos, 0.0, 1.0);
}