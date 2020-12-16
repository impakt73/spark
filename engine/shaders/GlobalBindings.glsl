layout(std430, set = 0, binding = 0) restrict buffer FrameMemoryBlockRW
{
    uint Data[];
} uFrameMemoryRW;

layout(std430, set = 0, binding = 0) restrict readonly buffer FrameMemoryBlockRO
{
    uint Data[];
} uFrameMemoryRO;

layout(push_constant) uniform PushConstantsBlock
{
    uint Constant0;
    uint Constant1;
    uint Constant2;
    uint Constant3;
} uPushConstants;

#define READ_BUFFER_UINT(inBuffer, inOffset) inBuffer.Data[inOffset]
#define READ_BUFFER_UINT2(inBuffer, inOffset) uvec2(inBuffer.Data[inOffset], inBuffer.Data[inOffset + 1])
#define READ_BUFFER_UINT3(inBuffer, inOffset) uvec3(inBuffer.Data[inOffset], inBuffer.Data[inOffset + 1], inBuffer.Data[inOffset + 2])
#define READ_BUFFER_UINT4(inBuffer, inOffset) uvec4(inBuffer.Data[inOffset], inBuffer.Data[inOffset + 1], inBuffer.Data[inOffset + 2], inBuffer.Data[inOffset + 3])

#define WRITE_BUFFER_UINT(inBuffer, inOffset, inData)  inBuffer.Data[inOffset]     = inData
#define WRITE_BUFFER_UINT2(inBuffer, inOffset, inData) inBuffer.Data[inOffset]     = inData.x;\
                                                       inBuffer.Data[inOffset + 1] = inData.y
#define WRITE_BUFFER_UINT3(inBuffer, inOffset, inData) inBuffer.Data[inOffset]     = inData.x;\
                                                       inBuffer.Data[inOffset + 1] = inData.y;\
                                                       inBuffer.Data[inOffset + 2] = inData.z
#define WRITE_BUFFER_UINT4(inBuffer, inOffset, inData) inBuffer.Data[inOffset]     = inData.x;\
                                                       inBuffer.Data[inOffset + 1] = inData.y;\
                                                       inBuffer.Data[inOffset + 2] = inData.z;\
                                                       inBuffer.Data[inOffset + 3] = inData.w

#define READ_BUFFER_FLOAT(inBuffer, inOffset) uintBitsToFloat(inBuffer.Data[inOffset])
#define READ_BUFFER_FLOAT2(inBuffer, inOffset) vec2(uintBitsToFloat(inBuffer.Data[inOffset]), uintBitsToFloat(inBuffer.Data[inOffset + 1]))
#define READ_BUFFER_FLOAT3(inBuffer, inOffset) vec3(uintBitsToFloat(inBuffer.Data[inOffset]), uintBitsToFloat(inBuffer.Data[inOffset + 1]), uintBitsToFloat(inBuffer.Data[inOffset + 2]))
#define READ_BUFFER_FLOAT4(inBuffer, inOffset) vec4(uintBitsToFloat(inBuffer.Data[inOffset]), uintBitsToFloat(inBuffer.Data[inOffset + 1]), uintBitsToFloat(inBuffer.Data[inOffset + 2]), uintBitsToFloat(inBuffer.Data[inOffset + 3]))

#define WRITE_BUFFER_FLOAT(inBuffer, inOffset, inData)  inBuffer.Data[inOffset]     = floatBitsToUint(inData)
#define WRITE_BUFFER_FLOAT2(inBuffer, inOffset, inData) inBuffer.Data[inOffset]     = floatBitsToUint(inData.x);\
                                                        inBuffer.Data[inOffset + 1] = floatBitsToUint(inData.y)
#define WRITE_BUFFER_FLOAT3(inBuffer, inOffset, inData) inBuffer.Data[inOffset]     = floatBitsToUint(inData.x);\
                                                        inBuffer.Data[inOffset + 1] = floatBitsToUint(inData.y);\
                                                        inBuffer.Data[inOffset + 2] = floatBitsToUint(inData.z)
#define WRITE_BUFFER_FLOAT4(inBuffer, inOffset, inData) inBuffer.Data[inOffset]     = floatBitsToUint(inData.x);\
                                                        inBuffer.Data[inOffset + 1] = floatBitsToUint(inData.y);\
                                                        inBuffer.Data[inOffset + 2] = floatBitsToUint(inData.z);\
                                                        inBuffer.Data[inOffset + 3] = floatBitsToUint(inData.w)

#define READ_BUFFER_MATRIX3(inBuffer, inOffset) mat3(uintBitsToFloat(inBuffer.Data[inOffset]),      uintBitsToFloat(inBuffer.Data[inOffset + 1]),  uintBitsToFloat(inBuffer.Data[inOffset + 2]),\
                                                     uintBitsToFloat(inBuffer.Data[inOffset + 3]),  uintBitsToFloat(inBuffer.Data[inOffset + 4]),  uintBitsToFloat(inBuffer.Data[inOffset + 5]),\
                                                     uintBitsToFloat(inBuffer.Data[inOffset + 6]),  uintBitsToFloat(inBuffer.Data[inOffset + 7]),  uintBitsToFloat(inBuffer.Data[inOffset + 8]))

#define READ_BUFFER_MATRIX4(inBuffer, inOffset) mat4(uintBitsToFloat(inBuffer.Data[inOffset]),      uintBitsToFloat(inBuffer.Data[inOffset + 1]),  uintBitsToFloat(inBuffer.Data[inOffset + 2]),  uintBitsToFloat(inBuffer.Data[inOffset + 3]),\
                                                     uintBitsToFloat(inBuffer.Data[inOffset + 4]),  uintBitsToFloat(inBuffer.Data[inOffset + 5]),  uintBitsToFloat(inBuffer.Data[inOffset + 6]),  uintBitsToFloat(inBuffer.Data[inOffset + 7]),\
                                                     uintBitsToFloat(inBuffer.Data[inOffset + 8]),  uintBitsToFloat(inBuffer.Data[inOffset + 9]),  uintBitsToFloat(inBuffer.Data[inOffset + 10]), uintBitsToFloat(inBuffer.Data[inOffset + 11]),\
                                                     uintBitsToFloat(inBuffer.Data[inOffset + 12]), uintBitsToFloat(inBuffer.Data[inOffset + 13]), uintBitsToFloat(inBuffer.Data[inOffset + 14]), uintBitsToFloat(inBuffer.Data[inOffset + 15]))

#define READ_BUFFER_MATRIX4X3(inBuffer, inOffset) mat4x3(uintBitsToFloat(inBuffer.Data[inOffset]),      uintBitsToFloat(inBuffer.Data[inOffset + 1]),  uintBitsToFloat(inBuffer.Data[inOffset + 2]),\
                                                         uintBitsToFloat(inBuffer.Data[inOffset + 3]),  uintBitsToFloat(inBuffer.Data[inOffset + 4]),  uintBitsToFloat(inBuffer.Data[inOffset + 5]),\
                                                         uintBitsToFloat(inBuffer.Data[inOffset + 6]),  uintBitsToFloat(inBuffer.Data[inOffset + 7]),  uintBitsToFloat(inBuffer.Data[inOffset + 8]),\
                                                         uintBitsToFloat(inBuffer.Data[inOffset + 9]),  uintBitsToFloat(inBuffer.Data[inOffset + 10]), uintBitsToFloat(inBuffer.Data[inOffset + 11]))