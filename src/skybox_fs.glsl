#version 330 core

uniform vec3 _colour;
uniform samplerCube cubemap;

in vec3 world_dir;

out vec4 color;

void main() {
    color = texture(cubemap, world_dir);
}
