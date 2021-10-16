layout(set = 1, binding = 0, rgba8) uniform image2D uImages[NUM_IMAGE_SLOTS];
layout(std430, set = 1, binding = 1) restrict buffer Buffers
{
    uint Data[];
} uBuffers[NUM_BUFFER_SLOTS];

layout(std430, set = 1, binding = 1) restrict readonly buffer BuffersRO
{
    uint Data[];
} uBuffersRO[NUM_BUFFER_SLOTS];

layout(std430, set = 1, binding = 1) restrict buffer Buffers64
{
    uint64_t Data[];
} uBuffers64[NUM_BUFFER_SLOTS];

layout(std430, set = 1, binding = 1) restrict buffer SignedBuffers64
{
    int64_t Data[];
} uSignedBuffers64[NUM_BUFFER_SLOTS];
