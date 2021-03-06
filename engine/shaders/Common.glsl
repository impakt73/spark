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