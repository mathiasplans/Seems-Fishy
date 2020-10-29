#include "common.h"

#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/System/Clock.hpp>
#include <SFML/Window/Event.hpp>
#include <imgui-SFML.h>
#include <imgui.h>

#include "cuda_wrapper.h"
#include "options.h"
#include "scenes/full_screen_opengl.h"

void handleEvents(sf::RenderWindow &window);

int main(int argc, const char **argv) {
  Options opt({std::next(argv), std::next(argv, argc)});
  opt.checkOptions();

  spdlog::info("Starting application");

  int gpuId = cuda::findBestDevice();
  cuda::gpuDeviceInit(gpuId);

  sf::RenderWindow window(sf::VideoMode(opt.width, opt.height), "SFML + CUDA",
                          sf::Style::Titlebar | sf::Style::Close);
  ImGui::SFML::Init(window);
  spdlog::info("SFML window created");

  FullScreenOpenGLScene scene(window);

  AppContext ctx;
  sf::Clock deltaClock;
  while (window.isOpen()) {
    ImGui::SFML::Update(window, deltaClock.restart());
    ctx.dtime = deltaClock.getElapsedTime().asSeconds();

    scene.update(ctx);

    ImGui::Begin("FPS");
    ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
    ImGui::Text("Frame %d", ctx.frame);
    ImGui::End();

    window.clear();
    scene.render(window);
    ImGui::SFML::Render(window);
    window.display();

    handleEvents(window);
    ctx.frame++;
  }

  spdlog::info("Shutting down");
  ImGui::SFML::Shutdown();

  return 0;
}

void handleEvents(sf::RenderWindow &window) {
  sf::Event event{};
  while (window.pollEvent(event)) {
    ImGui::SFML::ProcessEvent(event);

    if (event.type == sf::Event::Closed) {
      window.close();
    }

    if (event.type == sf::Event::KeyPressed) {
      if (event.key.code == sf::Keyboard::Escape) {
        window.close();
      }
    }
  }
}