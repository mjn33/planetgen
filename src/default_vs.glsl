#version 430

layout(std140) buffer CamBuffer {
    mat4 cam_matrix;
};

uniform mat4 obj_matrix;

in vec3 vert_pos;

void main() {
    gl_Position = cam_matrix * obj_matrix * vec4(vert_pos, 1.0);
}
