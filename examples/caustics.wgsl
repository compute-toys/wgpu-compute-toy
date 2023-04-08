#storage atomic_storage array<atomic<i32>>

// 2022 David A Roberts <https://davidar.io/>

// https://www.shadertoy.com/view/4djSRW
fn hash44(p: float4) -> float4 {
    var p4 = fract(p * float4(.1031, .1030, .0973, .1099));
    p4 = p4 + dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

const dt = 1.;
const n = float2(0., 1.);
const e = float2(1., 0.);
const s = float2(0., -1.);
const w = float2(-1., 0.);

fn A(fragCoord: float2) -> float4 {
    return passLoad(0, int2(fragCoord), 0);
}

fn B(fragCoord: float2) -> float4 {
    return passSampleLevelBilinearRepeat(1, fragCoord / float2(textureDimensions(screen)), 0.);
}

fn T(fragCoord: float2) -> float4 {
    return B(fragCoord - dt * B(fragCoord).xy);
}

@compute @workgroup_size(16, 16)
fn main_velocity(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let u = float2(id.xy) + 0.5;
    var r = T(u);
    r.x = r.x - dt * 0.25 * (T(u+e).z - T(u+w).z);
    r.y = r.y - dt * 0.25 * (T(u+n).z - T(u+s).z);

    if (time.frame < 3u) { r = float4(0.); }
    passStore(0, int2(id.xy), r);
}

@compute @workgroup_size(16, 16)
fn main_pressure(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let u = float2(id.xy) + 0.5;
    var r = A(u);
    r.z = r.z - dt * 0.25 * (A(u+e).x - A(u+w).x + A(u+n).y - A(u+s).y);

    let t = float(time.frame) / 120.;
    let o = float2(screen_size)/2. * (1. + .75 * float2(cos(t/15.), sin(2.7*t/15.)));
    r = mix(r, float4(0.5 * sin(dt * 2. * t) * sin(dt * t), 0., r.z, 1.), exp(-0.2 * length(u - o)));
    passStore(1, int2(id.xy), r);
}

@compute @workgroup_size(16, 16)
fn main_caustics(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    for (var i = 0; i < 25; i = i+1) {
        let h = hash44(float4(float2(id.xy), float(time.frame), float(i)));
        var p = float2(id.xy) + h.xy;
        let z = mix(.3, 1., h.z);
        let c = max(cos(z*6.2+float4(1.,2.,3.,4.)),float4(0.));
        let grad = 0.25 * float2(B(p+e).z - B(p+w).z, B(p+n).z - B(p+s).z);
        p = p + 1e5 * grad * z;
        p = fract(p / float2(screen_size)) * float2(screen_size);
        let idx = int(p.x) + int(p.y) * int(screen_size.x);
        atomicAdd(&atomic_storage[idx*4+0], int(c.x * 256.));
        atomicAdd(&atomic_storage[idx*4+1], int(c.y * 256.));
        atomicAdd(&atomic_storage[idx*4+2], int(c.z * 256.));
    }
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let idx = int(id.x) + int(id.y) * int(screen_size.x);
    let x = float(atomicLoad(&atomic_storage[idx*4+0]));
    let y = float(atomicLoad(&atomic_storage[idx*4+1]));
    let z = float(atomicLoad(&atomic_storage[idx*4+2]));
    var r = float3(x, y, z) / 256.;
    r = r * sqrt(r) / 5e3;
    r = r * float3(.5, .75, 1.);
    textureStore(screen, int2(id.xy), float4(r, 1.));
    atomicStore(&atomic_storage[idx*4+0], int(x * .9));
    atomicStore(&atomic_storage[idx*4+1], int(y * .9));
    atomicStore(&atomic_storage[idx*4+2], int(z * .9));
}
