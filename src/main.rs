extern crate cgmath;
#[macro_use]
extern crate glium;

use cgmath::prelude::*;
use cgmath::{Deg, Euler, Matrix4, Quaternion, Vector3};
use glium::Surface;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 3],
}

implement_vertex!(Vertex, position);

/// Generate a `Vec` containing a `size + 1` by `size + 1` grid of vertices.
fn gen_vertices(size: u16) -> Vec<Vertex> {
    let adj_size = size + 1;
    let mut vertices = Vec::new();
    let diff = 2.0 / size as f32;
    for i in 0..adj_size {
        for j in 0..adj_size {
            vertices.push(Vertex { position: [-1.0 + i as f32 * diff, -1.0 + j as f32 * diff, 0.0] });
        }
    }
    vertices
}

/// Generate indices to fill the specified range on a grid of size `size`.
fn gen_indices_range(x1: u16, y1: u16, x2: u16, y2: u16, size: u16) -> Vec<u16> {
    // TODO: maybe debug_assert! this instead?
    let (x1, x2) = if x1 < x2 { (x1, x2) } else { (x2, x1) };
    let (y1, y2) = if y1 < y2 { (y1, y2) } else { (y2, y1) };
    let adj_size = size + 1;
    let mut indices = Vec::new();
    for i in x1..x2 {
        for j in y1..y2 {
            // *--------*
            // | 2    - |
            // |    -   |
            // |  -     |
            // |-    1  |
            // *--------*
            let t1a = i * adj_size + j;
            let t1b = (i + 1) * adj_size + j;
            let t1c = i * adj_size + (j + 1);

            let t2a = (i + 1) * adj_size + j;
            let t2b = (i + 1) * adj_size + (j + 1);
            let t2c = i * adj_size + (j + 1);

            indices.push(t1a);
            indices.push(t1b);
            indices.push(t1c);
            indices.push(t2a);
            indices.push(t2b);
            indices.push(t2c);
        }
    }
    indices
}

/// Utility function for computing the index of a vertex with a given stride.
fn vert_off(x: u16, y: u16, stride: u16) -> u16 {
    x * stride + y
}

/// Generate the indices for a quad of the given size and the specified
/// edges "patched".
fn gen_indices(size: u16, l: bool, r: bool, t: bool, b: bool) -> Vec<u16> {
    let adj_size = size + 1;
    let vert_off = |x, y| vert_off(x, y, adj_size);
    let mut indices = gen_indices_range(1, 1, size - 1, size - 1, size);
    if l {
        for i in 1..adj_size-1 {
            if (i % 2) == 1 {
                // *
                // |-
                // |  -
                // |    -
                // |      -
                // |        *
                // |      -
                // |    -
                // |  -
                // |-
                // *
                let ta = vert_off(0, i - 1);
                let tb = vert_off(1, i);
                let tc = vert_off(0, i + 1);

                indices.push(ta);
                indices.push(tb);
                indices.push(tc);
            } else {
                //          *
                //         -|
                //       -  |
                //     -  2 |
                //   -      |
                // *--------*
                //   -      |
                //     -  1 |
                //       -  |
                //         -|
                //          *
                let t1a = vert_off(1, i - 1);
                let t1b = vert_off(1, i);
                let t1c = vert_off(0, i);

                let t2a = vert_off(0, i);
                let t2b = vert_off(1, i);
                let t2c = vert_off(1, i + 1);

                indices.push(t1a);
                indices.push(t1b);
                indices.push(t1c);
                indices.push(t2a);
                indices.push(t2b);
                indices.push(t2c);
            }
        }
    } else {
        if b {
            let ta = vert_off(1, 1);
            let tb = vert_off(0, 1);
            let tc = vert_off(0, 0);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        } else {
            indices.extend(gen_indices_range(0, 0, 1, 1, size));
        }
        indices.extend(gen_indices_range(0, 1, 1, size - 1, size));
        if t {
            let ta = vert_off(0, size);
            let tb = vert_off(0, size - 1);
            let tc = vert_off(1, size - 1);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        } else {
            indices.extend(gen_indices_range(0, size - 1, 1, size, size));
        }
    }

    if t {
        for j in 1..adj_size-1 {
            if (j % 2) == 1 {
                let ta = vert_off(j - 1, size);
                let tb = vert_off(j, size - 1);
                let tc = vert_off(j + 1, size);

                indices.push(ta);
                indices.push(tb);
                indices.push(tc);
            } else {
                let t1a = vert_off(j - 1, size - 1);
                let t1b = vert_off(j, size - 1);
                let t1c = vert_off(j, size);

                let t2a = vert_off(j, size);
                let t2b = vert_off(j, size - 1);
                let t2c = vert_off(j + 1, size - 1);

                indices.push(t1a);
                indices.push(t1b);
                indices.push(t1c);
                indices.push(t2a);
                indices.push(t2b);
                indices.push(t2c);
            }
        }
    } else {
        if l {
            let ta = vert_off(1, size - 1);
            let tb = vert_off(1, size);
            let tc = vert_off(0, size);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        }
        indices.extend(gen_indices_range(1, size - 1, size - 1, size, size));
        if r {
            let ta = vert_off(size, size);
            let tb = vert_off(size - 1, size);
            let tc = vert_off(size - 1, size - 1);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        }
    }

    if b {
        for j in 1..adj_size-1 {
            if (j % 2) == 1 {
                //
                //          *
                //        -   -
                //      -       -
                //    -           -
                //  -               -
                // *-----------------*
                let ta = vert_off(j + 1, 0);
                let tb = vert_off(j, 1);
                let tc = vert_off(j - 1, 0);

                indices.push(ta);
                indices.push(tb);
                indices.push(tc);
            } else {
                // *--------*--------*
                //  -       |       -
                //    - 1   |  2  -
                //      -   |   -
                //        - | -
                //          *
                let t1a = vert_off(j, 0);
                let t1b = vert_off(j, 1);
                let t1c = vert_off(j - 1, 1);

                let t2a = vert_off(j + 1, 1);
                let t2b = vert_off(j, 1);
                let t2c = vert_off(j, 0);

                indices.push(t1a);
                indices.push(t1b);
                indices.push(t1c);
                indices.push(t2a);
                indices.push(t2b);
                indices.push(t2c);
            }
        }
    } else {
        if l {
            let ta = vert_off(0, 0);
            let tb = vert_off(1, 0);
            let tc = vert_off(1, 1);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        }
        indices.extend(gen_indices_range(1, 0, size - 1, 1, size));
        if r {
            let ta = vert_off(size - 1, 1);
            let tb = vert_off(size - 1, 0);
            let tc = vert_off(size, 0);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        }
    }

    if r {
        let offset = adj_size * (adj_size - 2);
        for i in 1..adj_size-1 {
            if (i % 2) == 1 {
                //          *
                //         -|
                //       -  |
                //     -    |
                //   -      |
                // *        |
                //   -      |
                //     -    |
                //       -  |
                //         -|
                //          *
                let ta = vert_off(adj_size - 1, i + 1);
                let tb = vert_off(adj_size - 2, i);
                let tc = vert_off(adj_size - 1, i - 1);

                indices.push(ta);
                indices.push(tb);
                indices.push(tc);
            } else {
                // *
                // |-
                // |  -
                // |  2 -
                // |      -
                // *--------*
                // |      -
                // |  1 -
                // |  -
                // |-
                // *
                let t1a = vert_off(adj_size - 2, i);
                let t1b = vert_off(adj_size - 2, i - 1);
                let t1c = vert_off(adj_size - 1, i);

                let t2a = vert_off(adj_size - 2, i + 1);
                let t2b = vert_off(adj_size - 2, i);
                let t2c = vert_off(adj_size - 1, i);

                indices.push(t1a);
                indices.push(t1b);
                indices.push(t1c);
                indices.push(t2a);
                indices.push(t2b);
                indices.push(t2c);
            }
        }
    } else {
        if b {
            let ta = vert_off(size, 0);
            let tb = vert_off(size, 1);
            let tc = vert_off(size - 1, 1);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        } else {
            indices.extend(gen_indices_range(size - 1, 0, size, 1, size));
        }
        indices.extend(gen_indices_range(size - 1, 1, size, size - 1, size));
        if t {
            let ta = vert_off(size - 1, size - 1);
            let tb = vert_off(size, size - 1);
            let tc = vert_off(size, size);

            indices.push(ta);
            indices.push(tb);
            indices.push(tc);
        } else {
            indices.extend(gen_indices_range(size - 1, size - 1, size, size, size));
        }
    }
    indices
}

fn main() {
    use glium::DisplayBuild;
    let display = glium::glutin::WindowBuilder::new()
        .with_multisampling(8)
        .with_depth_buffer(24)
        .build_glium().unwrap();

    let vertices = gen_vertices(16);
    let indices = gen_indices(16, true, true, true, true);

    let vertex_buffer = glium::VertexBuffer::new(&display, &vertices).unwrap();
    let indices = glium::IndexBuffer::new(&display, glium::index::PrimitiveType::TrianglesList,
                                          &indices).unwrap();

    let vertex_shader_src = r#"
#version 140

in vec3 position;

uniform mat4 cam_matrix;

void main() {
    gl_Position = cam_matrix * vec4(position, 1.0);
}
"#;

    let fragment_shader_src = r#"
#version 140

out vec4 color;

void main() {
    color = vec4(1.0, 0.0, 0.0, 1.0);
}
"#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();
    let mut once = false;
    loop {
        let mut target = display.draw();
        target.clear_color_and_depth((0.8, 0.8, 0.8, 1.0), 1.0);

        let (width, height) = target.get_dimensions();
        //let aspect = height as f32 / width as f32;
        let aspect = width as f32 / height as f32;
        let fovy = Deg(90f32);
        let cam_pos = Vector3::new(0f32, 0f32, 1.5f32);
        let cam_rot = Quaternion::from(Euler {
            x: Deg(0f32),
            y: Deg(45f32),
            z: Deg(0f32),
        });
        let cam_perspective = cgmath::perspective(fovy, aspect, 0.1f32, 100f32);
        let cam_matrix =
            cam_perspective *
            Matrix4::from(cam_rot.invert()) *
            Matrix4::from_translation(-cam_pos);

        let cam_matrix = [
            [cam_matrix.x.x, cam_matrix.x.y, cam_matrix.x.z, cam_matrix.x.w],
            [cam_matrix.y.x, cam_matrix.y.y, cam_matrix.y.z, cam_matrix.y.w],
            [cam_matrix.z.x, cam_matrix.z.y, cam_matrix.z.z, cam_matrix.z.w],
            [cam_matrix.w.x, cam_matrix.w.y, cam_matrix.w.z, cam_matrix.w.w]
        ];

        if !once {
            println!("{:?}", cam_matrix);
            println!("=======");
            once = true;
        }

        let params = glium::DrawParameters {
            backface_culling: glium::draw_parameters::BackfaceCullingMode::CullClockwise,
            polygon_mode: glium::draw_parameters::PolygonMode::Line,
            depth: glium::Depth {
                test: glium::draw_parameters::DepthTest::IfLess,
                write: true,
                .. Default::default()
            },
            .. Default::default()
        };

        target.draw(&vertex_buffer, &indices, &program, &uniform! { cam_matrix: cam_matrix },
                    &params).unwrap();
        target.finish().unwrap();

        for ev in display.poll_events() {
            match ev {
                glium::glutin::Event::Closed => return,
                _ => std::thread::sleep_ms(1)
            }
        }
    }
}
