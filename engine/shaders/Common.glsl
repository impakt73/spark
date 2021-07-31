#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64                    : require

#extension GL_KHR_shader_subgroup_basic                  : require
#extension GL_KHR_shader_subgroup_vote                   : require
#extension GL_KHR_shader_subgroup_ballot                 : require
#extension GL_KHR_shader_subgroup_quad                   : require
#extension GL_KHR_shader_subgroup_arithmetic             : require

#define NUM_BUFFER_SLOTS  8
#define NUM_IMAGE_SLOTS   NUM_BUFFER_SLOTS
#define NUM_TEXTURE_SLOTS NUM_BUFFER_SLOTS

#include "GlobalBindings.glsl"
#include "GraphicsBindings.glsl"
#include "GraphBindings.glsl"

uint DynamicConstantOffset()
{
    return (uPushConstants.Constant0 & 0x00FFFFFF);
}

uint MaterialIndex()
{
    return ((uPushConstants.Constant0 >> 24) & 0xFF);
}

uint ShaderIoSlot0()
{
    return uPushConstants.Constant1;
}

uint ShaderIoSlot1()
{
    return uPushConstants.Constant2;
}

uint ShaderIoSlot2()
{
    return uPushConstants.Constant3;
}

// Static Constant Memory Offsets
const uint kCurrentTimeStaticOffset = 0;
const uint kSwapchainResolutionVec2StaticOffset = kCurrentTimeStaticOffset + 1;
const uint kSwapchainResolutionUvec2StaticOffset = kSwapchainResolutionVec2StaticOffset + 2;
const uint kProjMatrixStaticOffset = kSwapchainResolutionUvec2StaticOffset + 2;
const uint kViewMatrixStaticOffset = kProjMatrixStaticOffset + 16;
const uint kProjViewMatrixStaticOffset = kViewMatrixStaticOffset + 16;
