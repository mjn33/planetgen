#version 430

uniform mat4 _cam_matrix;
uniform mat4 _obj_matrix;

in vec3 vert_pos;

void main() {
    gl_Position = _cam_matrix * _obj_matrix * vec4(vert_pos, 1.0);
}
