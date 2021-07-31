layout(set = 1, binding = 0, rgba8) uniform image2D uImages[64];
layout(std430, set = 1, binding = 1) restrict buffer Buffers
{
    uint Data[];
} uBuffers[64];

layout(std430, set = 1, binding = 1) restrict buffer Buffers64
{
    uint64_t Data[];
} uBuffers64[64];

layout(std430, set = 1, binding = 1) restrict buffer SignedBuffers64
{
    int64_t Data[];
} uSignedBuffers64[64];