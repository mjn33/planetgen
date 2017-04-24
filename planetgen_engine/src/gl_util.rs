use gl;
use gl::types::*;
use std::fmt::Write;
use std::ptr;
use std::result::Result;
use std::string::FromUtf8Error;

pub struct Program {
    pub program: GLuint,
}

fn get_shader_info_log(shader: GLuint) -> Result<String, FromUtf8Error> {
    unsafe {
        let mut log_len = 0;
        gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut log_len);

        let mut log_arr: Vec<u8> = Vec::with_capacity(log_len as usize);
        log_arr.resize(log_len as usize, 0);

        gl::GetShaderInfoLog(shader, log_len, ptr::null_mut(),
                             log_arr.as_mut_ptr() as *mut GLchar);
        log_arr.pop();

        String::from_utf8(log_arr)
    }
}

fn get_program_info_log(program: GLuint) -> Result<String, FromUtf8Error> {
    unsafe {
        let mut log_len = 0;
        gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut log_len);

        let mut log_arr: Vec<u8> = Vec::with_capacity(log_len as usize);
        log_arr.resize(log_len as usize, 0);

        gl::GetProgramInfoLog(program, log_len, ptr::null_mut(),
                              log_arr.as_mut_ptr() as *mut GLchar);
        log_arr.pop();

        String::from_utf8(log_arr)
    }
}

impl Program {
    pub fn new(vs_src: &str, fs_src: &str) -> Result<Program, String> {
        unsafe {
            let vs = gl::CreateShader(gl::VERTEX_SHADER);
            let fs = gl::CreateShader(gl::FRAGMENT_SHADER);
            gl::ShaderSource(vs, 1, &(vs_src.as_ptr() as *const GLchar), &(vs_src.len() as GLint));
            gl::ShaderSource(fs, 1, &(fs_src.as_ptr() as *const GLchar), &(fs_src.len() as GLint));

            gl::CompileShader(vs);
            gl::CompileShader(fs);

            let mut vs_status = 0;
            let mut fs_status = 0;
            gl::GetShaderiv(vs, gl::COMPILE_STATUS, &mut vs_status);
            gl::GetShaderiv(fs, gl::COMPILE_STATUS, &mut fs_status);

            if vs_status == gl::FALSE as GLint || fs_status == gl::FALSE as GLint {
                let mut msg = String::new();
                if vs_status == gl::FALSE as GLint {
                    let log = get_shader_info_log(vs);

                    writeln!(msg, "Vertex Shader compile log: ").unwrap();
                    match log {
                        Ok(log) => writeln!(msg, "{}", log).unwrap(),
                        Err(_) => writeln!(msg, "[log not valid UTF-8]").unwrap(),
                    }
                }
                if fs_status == gl::FALSE as GLint {
                    let log = get_shader_info_log(fs);

                    writeln!(msg, "Fragment Shader compile log: ").unwrap();
                    match log {
                        Ok(log) => writeln!(msg, "{}", log).unwrap(),
                        Err(_) => writeln!(msg, "[log not valid UTF-8]").unwrap(),
                    }
                }

                return Err(msg);
            }

            let program = gl::CreateProgram();
            gl::AttachShader(program, vs);
            gl::AttachShader(program, fs);
            gl::LinkProgram(program);

            let mut link_status = 0;
            gl::GetProgramiv(program, gl::LINK_STATUS, &mut link_status);

            if link_status == gl::FALSE as GLint {
                let mut msg = String::new();
                let log = get_program_info_log(program);

                writeln!(msg, "Program link log: ").unwrap();

                match log {
                    Ok(log) => writeln!(msg, "{}", log).unwrap(),
                    Err(_) => writeln!(msg, "[log not valid UTF-8]").unwrap(),
                }

                return Err(msg);
            }
            Ok(Program {
                program,
            })
        }
    }

    /// Returns a vector of tuples containing the name of the uniform and its
    /// location.
    pub fn uniforms(&self) -> Vec<(String, GLint)> {
        unsafe {
            let program = self.program;
            let properties = [gl::BLOCK_INDEX, gl::NAME_LENGTH, gl::LOCATION];
            let mut num_resources = 0;
            let mut uniforms = Vec::new();
            gl::GetProgramInterfaceiv(program, gl::UNIFORM, gl::ACTIVE_RESOURCES, &mut num_resources);
            for i in 0..num_resources {
                let mut values = [0, 0, 0];
                gl::GetProgramResourceiv(program, gl::UNIFORM, i as GLuint, 3, properties.as_ptr(), 3, ptr::null_mut(), values.as_mut_ptr());

                if values[0] != -1 {
                    continue;
                }

                let name_len = values[1];
                let mut name_arr: Vec<u8> = Vec::with_capacity(name_len as usize);
                name_arr.resize(name_len as usize, 0);

                gl::GetProgramResourceName(
                    program, gl::UNIFORM, i as GLuint, name_len, ptr::null_mut(),
                    name_arr.as_mut_ptr() as *mut GLchar);
                name_arr.pop();

                let name = String::from_utf8(name_arr).expect("Uniform name isn't value UTF-8!");

                uniforms.push((name, values[2]));
            }

            uniforms
        }
    }
}
