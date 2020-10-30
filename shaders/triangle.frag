#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 pos;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(pos, 0.0, 1.0);
}

//return type for hit function
struct MarchHit {
  float dist;
  vec3 normal;
  vec3 color;
};

MarchHit sphere(vec3 spherePosition, vec3 rayPosition, int radius, vec3 color){
    float dist = distance(spherePosition, rayPosition) - radius;
    vec3 normal =  normalize(rayPosition - spherePosition);
    MarchHit hit;
    hit.dist = dist;
    hit.normal = normal;
    hit.color = color;

    return hit;
}