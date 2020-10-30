#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 pos;

layout(location = 0) out vec4 outColor;

//return type for hit function
struct MarchHit {
  float dist;
  vec3 normal;
  vec3 color;
  vec3 pos;
};

MarchHit sphere(vec3 spherePosition, vec3 rayPosition, float radius, vec3 color) {
    float dist = distance(spherePosition, rayPosition) - radius;
    vec3 normal =  normalize(rayPosition - spherePosition);
    MarchHit hit;
    hit.dist = dist;
    hit.normal = normal;
    hit.color = color;
    hit.pos = rayPosition;

    return hit;
}

MarchHit march(vec3 position, vec3 dir) {
    MarchHit hit = sphere(vec3(1.0, 1.0, -3.0), position, 1.0, vec3(1.0));

    while (hit.dist > 0.001) {
        position = position + dir * hit.dist;
        hit = sphere(vec3(1.0, 1.0, -3.0), position, 1.0, vec3(1.0));

        if (hit.dist > 6.0) {
            MarchHit miss;
            miss.dist = 100000.0;
            miss.normal = vec3(0.0);
            miss.color = vec3(0.0);

            return miss;
        }
    }

    return hit;
}

void main() {
    vec3 source = vec3(0.0, 0.0, 1.0);
    vec3 pos3d = vec3(pos, 0.0);
    vec3 dir = normalize(pos3d - source);

    vec3 lightPos = vec3(3.0);

    MarchHit hit = march(pos3d, dir);

    vec3 lightDir = normalize(hit.pos - lightPos);

    vec3 col = hit.color * (dot(-lightDir, hit.normal));

    outColor = vec4(col, 1.0);
}
