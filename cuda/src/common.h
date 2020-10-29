#pragma once

#include <spdlog/spdlog.h>

union Color4 // 4 bytes = 4 chars = 1 float
{
  float c;
  unsigned char components[4];
};

struct Pixel {
  float x, y;
  Color4 color;
};

struct AppContext {
  size_t frame = 0;
  float dtime  = 0.f;
};
