#version 330 core

uniform vec3 _colour;
uniform float sun_radius;

in vec2 coord;

out vec4 color;

void main() {
    float a = clamp(length(coord) - 0.5 * sun_radius, 0.0, 0.5 * sun_radius) / (0.5 * sun_radius);
    color = mix(vec4(1.0, 0.9, 0.7, 1.0),
                vec4(1.0, 0.9, 0.7, 0.0),
                a);
}
