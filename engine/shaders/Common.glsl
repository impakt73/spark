#include "GlobalBindings.glsl"
#include "GraphicsBindings.glsl"

uint DynamicConstantOffset()
{
    return (uPushConstants.Constant0 & 0x00FFFFFF);
}

uint MaterialIndex()
{
    return ((uPushConstants.Constant0 >> 24) & 0xFF);
}