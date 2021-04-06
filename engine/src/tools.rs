use std::{error::Error, fmt, fs, path::Path};

use shaderc::{
    CompileOptions, Compiler, EnvVersion, IncludeType, OptimizationLevel, ResolvedInclude,
    ShaderKind, TargetEnv,
};

/// The following shaders are available through the crate to allow external code to compile compatible spv
const COMMON_GLSL_STR: &str = include_str!("../shaders/Common.glsl");
const GLOBAL_BINDINGS_GLSL_STR: &str = include_str!("../shaders/GlobalBindings.glsl");
const GRAPH_BINDINGS_GLSL_STR: &str = include_str!("../shaders/GraphBindings.glsl");
const GRAPHICS_BINDINGS_GLSL_STR: &str = include_str!("../shaders/GraphicsBindings.glsl");

#[derive(Debug)]
enum ShaderCompilerError {
    CreateCompilerError,
    CompileOptionsError,
}

impl fmt::Display for ShaderCompilerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#?}", self)
    }
}

impl Error for ShaderCompilerError {}

/// Handles a load operation for a shader file accessed by a relative path
fn load_relative_file(include_dir: &Option<String>, path: &str) -> Option<String> {
    let path = if let Some(include_dir) = &include_dir {
        Path::new(&include_dir).join(path)
    } else {
        Path::new(path).to_path_buf()
    };
    std::fs::read_to_string(&path).ok()
}

/// Handles a load operation for a shader file accessed by a standard path
///
/// These files are stored inside this compiled binary and do not come from disk
fn load_standard_file(path: &str) -> Option<String> {
    match path {
        "Common.glsl" => Some(COMMON_GLSL_STR.to_owned()),
        "GlobalBindings.glsl" => Some(GLOBAL_BINDINGS_GLSL_STR.to_owned()),
        "GraphBindings.glsl" => Some(GRAPH_BINDINGS_GLSL_STR.to_owned()),
        "GraphicsBindings.glsl" => Some(GRAPHICS_BINDINGS_GLSL_STR.to_owned()),
        _ => None,
    }
}

/// Returns true if the provided extension matches a supported GLSL shader type
fn is_shader_ext(ext: &str) -> bool {
    matches!(ext, "vert" | "frag" | "comp")
}

/// Helper structure for shader compilation
///
/// This compiler can be used to compile shaders that are compatible with the render graph
/// shader infrastructure.
pub struct ShaderCompiler<'a> {
    compile_options: CompileOptions<'a>,
    compiler: Compiler,
}

impl<'a> ShaderCompiler<'a> {
    /// Creates a new compiler with an optional include directory
    pub fn new(include_dir: Option<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let compiler = Compiler::new().ok_or(ShaderCompilerError::CreateCompilerError)?;
        let mut compile_options =
            CompileOptions::new().ok_or(ShaderCompilerError::CompileOptionsError)?;
        compile_options.set_include_callback(move |name, inc_type, _parent_name, _depth| {
            let file_contents = match inc_type {
                IncludeType::Relative => load_relative_file(&include_dir, name),
                IncludeType::Standard => load_standard_file(name),
            };
            if let Some(content) = file_contents {
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
        compile_options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_2 as u32);
        compile_options.set_optimization_level(OptimizationLevel::Performance);
        compile_options.set_warnings_as_errors();

        Ok(Self {
            compile_options,
            compiler,
        })
    }

    /// Compiles a single shader at the provided input path and stores the result at the output path
    pub fn compile_shader(
        &mut self,
        input_path: &str,
        output_path: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let source = fs::read_to_string(&input_path)?;
        let compile_result = self.compiler.compile_into_spirv(
            &source,
            ShaderKind::InferFromSource,
            input_path,
            "main",
            Some(&self.compile_options),
        )?;

        fs::write(output_path, compile_result.as_binary_u8())?;
        Ok(())
    }

    /// Compiles all shaders in the provided input directory and writes the compiled results into the output directory
    pub fn compile_shaders(
        &mut self,
        input_dir: &str,
        output_dir: &str,
    ) -> Result<(), Box<dyn Error>> {
        for file in fs::read_dir(input_dir)? {
            let file = file?;
            let path = file.path();
            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if is_shader_ext(ext.to_str().unwrap()) {
                        let filename = path.file_name().unwrap().to_str().unwrap().to_string();
                        let out_path = Path::new(output_dir).join(filename).with_extension("spv");
                        self.compile_shader(path.to_str().unwrap(), out_path.to_str().unwrap())?;
                    }
                }
            }
        }
        Ok(())
    }
}
