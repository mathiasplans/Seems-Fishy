#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 pos;
layout(location = 2) in float time;

layout(location = 0) out vec4 outColor;

struct Ray {
    vec3 pos;
    vec3 dir;
    vec3 color;
    float energy;
};

struct Material {
    vec3 color;
    vec3 diffuse;
    vec3 shininess;
};

//return type for hit function
struct MarchHit {
  float dist;
  vec3 normal;
  vec3 pos;
  Material material;
};

MarchHit sphere(vec3 spherePosition, Ray ray, float radius, vec3 color, Material material) {
    float dist = distance(spherePosition, ray.pos) - radius;
    vec3 normal =  normalize(ray.pos - spherePosition);
    MarchHit hit;
    hit.dist = dist;
    hit.normal = normal;
    hit.material = material;
    //hit.color = mix(ray.color * material.color, material.color * material.ambient, 0.1);
    hit.pos = ray.pos;

    return hit;
}

MarchHit plane(vec3 planePosition, Ray ray, vec3 normal, vec3 color, Material material){
  vec3 planeToRay = ray.pos - planePosition;;
  float dist = length(dot(normal, planeToRay));

  MarchHit hit;
  hit.dist = dist;
  hit.normal = normalize(normal);
  hit.material = material;
  //hit.material.color = color;
  hit.pos = ray.pos;

  return hit;
}

MarchHit water(vec3 waterPosition, Ray ray, float amplitude, vec3 normal, vec3 color, Material material){
    vec3 planePosition = waterPosition;
    if(ray.pos.y > waterPosition.y){
        planePosition.y += amplitude;
        return plane(planePosition, ray, normal, color,material);
    }

    else{
         planePosition.y -= amplitude;
        return plane(planePosition, ray, normal, color,material);
    }
}

Material createMaterial(vec3 color, vec3 diffuse, vec3 shininess) {
    Material newMaterial;

    newMaterial.color = color;
    newMaterial.diffuse = diffuse;
    newMaterial.shininess = shininess;

    return newMaterial;
}

MarchHit smallest(Ray ray) {
    Material basic1 = createMaterial(vec3(0.7, 0.0, 0.3), vec3(0.1), vec3(0.0));
    Material basic2 = createMaterial(vec3(0.1, 0.8, 0.1), vec3(0.1), vec3(0.0));
    Material wall1 = createMaterial(vec3(1.0), vec3(0.1), vec3(0.0));
    Material wall2 = createMaterial(vec3(0.6, 0.7, 0.2), vec3(0.1), vec3(0.0));
    Material wall3 = createMaterial(vec3(0.0, 0.0, 1.0), vec3(0.1), vec3(0.0));

    MarchHit hits[] = {
        sphere(vec3(1.0, 1.0, -3.0), ray, 1.0, vec3(1.0, 0.0, 1.0), basic1),
        // plane(vec3(0.0, -10.0, -10.0), position, vec3(0.0, 1.0, 1.0), vec3(1.0, 0.0, 1.0))
        sphere(vec3(6.0, 4.0, -6.0), ray, 1.0, vec3(1.0, 1.0, 0.0), basic2),

        plane(vec3(10.0, 0.0, 0.0), ray, vec3(-1.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), wall1),
        plane(vec3(0.0, 10.0, 0.0), ray, vec3(0.0, -1.0, 0.0), vec3(1.0, 0.0, 0.0), wall2),
        plane(vec3(0.0, 0.0, 10.0), ray, vec3(0.0, 0.0, -1.0), vec3(1.0, 1.0, 1.0), wall3),
        plane(vec3(-10.0, 0.0, 0.0), ray, vec3(1.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0), wall1),
        plane(vec3(0.0, -10.0, 0.0), ray, vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 1.0), wall2),
        plane(vec3(0.0, 0.0, -10.0), ray, vec3(0.0, 0.0, 1.0), vec3(1.0, 1.0, 1.0), wall3)
    };

    MarchHit bestHit = hits[0];
    for (int i = 1; i < 8; ++i) {
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
        if (hit.dist > 100.0) {
            MarchHit miss;
            miss.dist = 100000.0;
            miss.normal = vec3(0.0);
            miss.material.color = vec3(0.0);

            return miss;
        }
    }

    return hit;
}

MarchHit multi_march(Ray ray, int jumps, vec3 lightPos) {
    MarchHit first = march(ray);
    MarchHit current = first;

    MarchHit hits[10];
    hits[0] = current;

    int jump;
    for (jump = 1; jump < 10; ++jump) {
        // Get new direction
        ray.dir = reflect(ray.dir, current.normal);

        // Go away a bit
        current.pos += ray.dir * 0.01;

        // Do the march
        MarchHit current = march(ray);

        // Miss
        if (current.normal == vec3(0.0))
            break;

        hits[jump] = current;
    }

    // Mixing color
    for (int a = jump; a > 0; --a){
        vec3 lightDir = normalize(hits[a - 1].pos - lightPos);
        hits[a - 1].material.color *= dot(-lightDir, hits[a - 1].normal);

        ray.color = ray.color * hits[a-1].material.color;
        // ray.color = mix(ray.color * hits[a-1].material.color , ray.color, ray.energy);
        // ray.energy *= 0.7;
    }

    first.material.color = ray.color;

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
    ray.color = vec3(1,1,1);
    MarchHit hit = multi_march(ray, 10, lightPos);

    vec3 lightDir = normalize(hit.pos - lightPos);

    vec3 col = hit.material.color; //* (dot(-lightDir, hit.normal));

    outColor = vec4(col, 1.0);
}
