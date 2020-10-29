INCLUDE = -Iinc -Iglm -Iglfw/include
CFLAGS = -std=c++17 -O2 $(INCLUDE)
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

build/VulkanTest: build src/main.cpp libglfw3.a
    g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

build:
	@mkdir build

.PHONY: test clean
test: build/VulkanTest
    ./VulkanTest

clean:
    rm -f VulkanTest
