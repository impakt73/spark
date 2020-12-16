#version 450

#pragma shader_stage(vertex)

layout(location = 0) in vec2 InPos;
layout(location = 1) in vec2 InTexCoord;
layout(location = 2) in uint InColor;

layout(location = 0) out vec2 vTexCoord;
layout(location = 1) out vec4 vColor;

#include "Common.glsl"

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    // TODO: Deal with vk clip space on the CPU side
    const mat4 clipMatrix = mat4(
        1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.5, 0.0,
        0.0, 0.0, 0.5, 1.0
    );

    const mat4 mvpMatrix = clipMatrix * READ_BUFFER_MATRIX4(uFrameMemoryRO, DynamicConstantOffset());

    vTexCoord = InTexCoord;

    vColor = vec4(
        (InColor & 0xFF) / 255.0,
        ((InColor >> 8) & 0xFF) / 255.0,
        ((InColor >> 16) & 0xFF) / 255.0,
        ((InColor >> 24) & 0xFF) / 255.0
    );

    gl_Position = mvpMatrix * vec4(InPos.x, InPos.y, 0.0, 1.0);
}