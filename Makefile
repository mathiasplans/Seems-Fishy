OUTPUT = build/bin/Debug/vert.spv build/bin/Debug/frag.spv

all: $(OUTPUT)

build/bin/Debug:
	mkdir $@

build/bin/Debug/%.spv: shaders/triangle.% build/bin/Debug
	glslc.exe $< -o $@

clean:
	rm -rf $(OUTPUT)