#version 450

#pragma shader_stage(vertex)

layout(location = 0) out vec2 vTexCoord;

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    const float idOverTwo = float(gl_VertexIndex / 2);
    const float idModTwo = float(gl_VertexIndex % 2);

    vTexCoord = vec2(idOverTwo * 2.0, idModTwo * 2.0);
    gl_Position = vec4(idOverTwo * 4.0 - 1.0, idModTwo * 4.0 - 1.0, 0.0, 1.0);
}