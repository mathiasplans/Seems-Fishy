OUTPUT = vert.spv frag.spv

all: $(OUTPUT)

%.spv: shaders/triangle.%
	glslc.exe $< -o $@

clean:
	rm -rf $(OUTPUT)