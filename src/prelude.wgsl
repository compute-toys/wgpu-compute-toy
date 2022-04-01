type int = i32;
type uint = u32;
type float = f32;

type int2 = vec2<i32>;
type int3 = vec3<i32>;
type int4 = vec4<i32>;
type uint2 = vec2<u32>;
type uint3 = vec3<u32>;
type uint4 = vec4<u32>;
type float2 = vec2<f32>;
type float3 = vec3<f32>;
type float4 = vec4<f32>;

struct Screen { size: uint2 };
struct Time { frame: uint, elapsed: float };
struct Mouse { pos: uint2, click: int };
struct StorageBuffer { data: array<atomic<i32>> };

@group(0) @binding(0) var<uniform> screen: Screen;
@group(0) @binding(1) var<uniform> time: Time;
@group(0) @binding(2) var<uniform> mouse: Mouse;
@group(0) @binding(3) var col: texture_storage_2d<rgba16float,write>;
@group(0) @binding(4) var<storage,read_write> buf: StorageBuffer;
@group(0) @binding(5) var tex: texture_2d_array<f32>;
@group(0) @binding(6) var texs: texture_storage_2d_array<rgba16float,write>;
@group(0) @binding(7) var nearest: sampler;
@group(0) @binding(8) var bilinear: sampler;
