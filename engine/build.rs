use shaderc::{CompileOptions, Compiler, ResolvedInclude, ShaderKind};
use std::path::{Path, PathBuf};
use std::{env, fs};
use walkdir::WalkDir;

/// Compiles the requested GLSL shaders into SPV and puts them in the build output directory
fn compile_shaders(shaders_dir: &str, shaders: &[&str]) {
    let mut compiler = Compiler::new().expect("Failed to create compiler");

    let mut compile_options = CompileOptions::new().unwrap();
    compile_options.set_include_callback(move |name, _inc_type, _parent_name, _depth| {
        let path = Path::new(shaders_dir).join(name);
        if let Ok(content) = std::fs::read_to_string(&path) {
            Ok(ResolvedInclude {
                resolved_name: String::from(name),
                content,
            })
        } else {
            Err(format!(
                "Failed to load included shader code from {}.",
                name
            ))
        }
    });

    for shader in shaders {
        let path = Path::new(shaders_dir).join(shader);

        println!("{}", env::current_dir().unwrap().to_string_lossy());
        println!("{}", path.to_string_lossy());
        let source = fs::read_to_string(&path).unwrap();
        let compile_result = compiler
            .compile_into_spirv(
                &source,
                ShaderKind::InferFromSource,
                shader,
                "main",
                Some(&compile_options),
            )
            .unwrap();

        let out_filename = shader.to_string() + ".spv";
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap()).join(out_filename);
        fs::write(out_path, compile_result.as_binary_u8()).unwrap();
    }
}

fn main() {
    for entry in WalkDir::new("foo").into_iter().filter_map(|e| e.ok()) {
        println!("cargo:rerun-if-changed={}", entry.path().to_string_lossy());
    }

    compile_shaders("shaders", &["ImguiTriangle.vert", "ImguiTriangle.frag"]);
}
