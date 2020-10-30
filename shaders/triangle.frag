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

MarchHit plane(vec3 planePosition, vec3 rayPosition, vec3 planeVector1, vec3 planeVector2, vec3 color){
  vec3 planeToRay = rayPosition - planePosition;
  vec3 normal = cross(planeVector1, planeVector2);
  float dist = length(dot(normal, planeToRay)) / length(normal);

  MarchHit hit;
  hit.dist = dist;
  hit.normal = normalize(normal);
  hit.color = color;
  hit.pos = rayPosition;

  return hit;
}

MarchHit smallest(vec3 position, vec3 dir) {
    MarchHit hits[] = {
        sphere(vec3(1.0, 1.0, -3.0), position, 1.0, vec3(1.0)),
        plane(vec3(0.0, 0.0, -10.0), position, vec3(1.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 1.0))
    };

    MarchHit bestHit = hits[0];
    for (int i = 1; i < 2; ++i) {
        MarchHit candidate = hits[i];

        if (bestHit.dist > candidate.dist)
            bestHit = candidate;
    }

    return bestHit;
}

MarchHit march(vec3 position, vec3 dir) {
    MarchHit hit = smallest(position, dir);

    while (hit.dist > 0.001) {
        position = position + dir * hit.dist;
        hit = smallest(position, dir);

        if (hit.dist > 20.0) {
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
