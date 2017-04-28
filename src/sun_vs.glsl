#version 330 core

uniform vec3 _cam_pos;
uniform mat4 _cam_matrix;
uniform mat4 _obj_matrix;
uniform float sun_radius;

in vec3 vpos;
in vec3 vnorm;
in vec3 vcolour;

out vec2 coord;

void main() {
    vec3 world_vpos = (_obj_matrix * vec4(vpos * sun_radius, 1.0)).xyz;
    gl_Position = _cam_matrix * vec4(world_vpos, 1.0);
    coord = vpos.xy * sun_radius;
}
