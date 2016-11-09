#version 140

in vec3 _mat_colour;

out vec4 color;

void main() {
    color = vec4(_mat_colour, 1.0);
}
