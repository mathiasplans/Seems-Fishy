#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 pos;

layout(location = 0) out vec4 outColor;

struct Ray{
    vec3 pos;
    vec3 dir;
    vec3 color;
    float energy;
};

//return type for hit function
struct MarchHit {
  float dist;
  vec3 normal;
  vec3 color;
  vec3 pos;
};

MarchHit sphere(vec3 spherePosition, Ray ray, float radius, vec3 color) {
    float dist = distance(spherePosition, ray.pos) - radius;
    vec3 normal =  normalize(ray.pos - spherePosition);
    MarchHit hit;
    hit.dist = dist;
    hit.normal = normal;
    hit.color = color;
    hit.pos = ray.pos;

    return hit;
}

MarchHit plane(vec3 planePosition, Ray ray, vec3 normal, vec3 color){
  vec3 planeToRay = ray.pos - planePosition;;
  float dist = length(dot(normal, planeToRay));

  MarchHit hit;
  hit.dist = dist;
  hit.normal = normalize(normal);
  hit.color = color;
  hit.pos = ray.pos;

  return hit;
}

MarchHit smallest(Ray ray) {
    MarchHit hits[] = {
        sphere(vec3(1.0, 1.0, -3.0), ray, 1.0, vec3(1.0, 0.0, 1.0)),
        // plane(vec3(0.0, -10.0, -10.0), position, vec3(0.0, 1.0, 1.0), vec3(1.0, 0.0, 1.0))
        sphere(vec3(3.0, 3.0, -4.0), ray, 1.0, vec3(1.0, 1.0, 0.0))
    };

    MarchHit bestHit = hits[0];
    for (int i = 1; i < 2; ++i) {
        MarchHit candidate = hits[i];

        if (bestHit.dist > candidate.dist)
            bestHit = candidate;
    }

    return bestHit;
}

MarchHit march(Ray ray) {
    MarchHit hit = smallest(ray);

    while (hit.dist > 0.001) {
        ray.pos = ray.pos + ray.dir * hit.dist;
        hit = smallest(ray);

        // Too far, stop
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

MarchHit multi_march(Ray ray, int jumps) {
    MarchHit first = march(ray);
    MarchHit current = first;
    for (int i = 1; i < jumps; ++i) {
        // Get new direction
        ray.dir = reflect(ray.dir, current.normal);

        // Do the march
        MarchHit current = march(ray);

        // Add the color
        first.color = mix(first.color, current.color, 0.2);
    }

    return first;
}

void main() {
    vec3 source = vec3(0.0, 0.0, 1.0);
    vec3 pos3d = vec3(pos, 0.0);
    vec3 dir = normalize(pos3d - source);

    vec3 lightPos = vec3(3.0);

    Ray ray;
    ray.pos = pos3d;
    ray.dir = dir;
    MarchHit hit = multi_march(ray, 3);

    vec3 lightDir = normalize(hit.pos - lightPos);

    vec3 col = hit.color * (dot(-lightDir, hit.normal));

    outColor = vec4(col, 1.0);
}
