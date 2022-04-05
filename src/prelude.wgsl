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

struct Time { frame: uint, elapsed: float };
struct Mouse { pos: uint2, click: int };

@group(0) @binding(1) var<uniform> time: Time;
@group(0) @binding(2) var<uniform> mouse: Mouse;
@group(0) @binding(3) var<uniform> _keyboard: array<vec4<u32>,2>;
@group(0) @binding(4) var screen: texture_storage_2d<rgba16float,write>;
@group(0) @binding(5) var<storage,read_write> atomic_storage: array<atomic<i32>>;
@group(0) @binding(6) var pass_in: texture_2d_array<f32>;
@group(0) @binding(7) var pass_out: texture_storage_2d_array<rgba16float,write>;
@group(0) @binding(8) var nearest: sampler;
@group(0) @binding(9) var bilinear: sampler;
@group(0) @binding(10) var channel0: texture_2d<f32>;

fn keyDown(keycode: uint) -> bool {
    return ((_keyboard[keycode / 128u][(keycode % 128u) / 32u] >> (keycode % 32u)) & 1u) == 1u;
}
