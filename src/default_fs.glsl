#version 330 core

//in vec3 _mat_colour;
in vec3 world_vpos;
in vec3 world_vnorm;
in vec3 world_view_dir;

out vec4 color;

void main() {
    vec3 light_dir = vec3(0.0, 0.0, -1.0); // TODO
    vec3 ambient_colour = vec3(0.0, 0.0, 0.0);
    vec3 diffuse_colour = vec3(0.8, 0.2, 0.2);
    vec3 specular_colour = vec3(0.2, 0.2, 0.2);
    float n_specular = 5.0;

    float cos_theta = clamp(dot(world_vnorm, light_dir), 0, 1);

    vec3 view_dir = normalize(world_view_dir);
    vec3 reflect_dir = reflect(-light_dir, world_vnorm);
    float cos_phi = clamp(dot(view_dir, reflect_dir), 0, 1);

    //color = vec4(0.0, 0.0, 0.0, 1.0);
    color = vec4(ambient_colour
                 + cos_theta * diffuse_colour
                 + pow(cos_phi, n_specular) * specular_colour, 1.0);
}
