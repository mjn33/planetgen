#version 330 core

uniform vec3 _cam_pos;
uniform mat4 _cam_matrix;
uniform mat4 _obj_matrix;

in vec3 vpos;
in vec3 vnorm;
in vec3 vcolour;

out vec3 world_vpos;
out vec3 world_vnorm;
out vec3 colour;
out vec3 world_view_dir;

void main() {
    world_vpos = (_obj_matrix * vec4(vpos, 1.0)).xyz;
    world_vnorm = vnorm; // TODO
    //world_vnorm = (_obj_rot_matrix * vec4(vnorm, 1.0)).xyz;
    world_view_dir = _cam_pos - world_vpos;
    gl_Position = _cam_matrix * vec4(world_vpos, 1.0);
    colour = vcolour;
}
