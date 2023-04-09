#storage atomic_storage array<atomic<i32>>

alias float4x4 = mat4x4<f32>;

fn rotXW(t: float) -> float4x4 {
    return float4x4(
        1.0, 0.0, 0.0, 0.0,
        0.0, cos(t), sin(t), 0.0,
        0.0, - sin(t), cos(t), 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}
fn rotXY(t: float) -> float4x4 {
    return float4x4(
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, cos(t), sin(t),
        0.0, 0.0, - sin(t), cos(t)
    );
}
fn rotXZ(t: float) -> float4x4 {
    return float4x4(
        1.0, 0.0, 0.0, 0.0,
        0.0, cos(t), 0.0, sin(t),
        0.0, 0.0, 1.0, 0.0,
        0.0, - sin(t), 0.0, cos(t)
    );
}
fn rotYZ(t: float) -> float4x4 {
    return float4x4(
        cos(t), 0.0, 0.0, sin(t),
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        - sin(t), 0.0, 0.0, cos(t)
    );
}
fn rotYW(t: float) -> float4x4 {
    return float4x4(
        cos(t), 0.0, sin(t), 0.0,
        0.0, 1.0, 0.0, 0.0,
        - sin(t), 0.0, cos(t), 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}
fn rotZW(t: float) -> float4x4 {
    return float4x4(
        cos(t), sin(t), 0.0, 0.0,
        - sin(t), cos(t), 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
}

fn matrot(p: float4) -> float4 {
    let t = float(time.frame)/240.;
    let m = rotZW(t) * rotXW(t*1.3) * rotYW(t*1.6);
    return m * p;
}

fn toScreen(p : float4, R : float2) -> float2 {
    let q = matrot(p - 0.5).xy;
    return (0.5*q+0.5) * (R) * float2(R.y/R.x,1.) + float2(0.25*R.x,0.);
}

@compute @workgroup_size(16, 16)
fn clear(@builtin(global_invocation_id) id: uint3) {
    let screen_size = int2(textureDimensions(screen));
    let idx0 = int(id.x) + int(screen_size.x * int(id.y));

    atomicStore(&atomic_storage[idx0*4+0], 0);
    atomicStore(&atomic_storage[idx0*4+1], 0);
    atomicStore(&atomic_storage[idx0*4+2], 0);
    atomicStore(&atomic_storage[idx0*4+3], 0);
}

@compute @workgroup_size(16, 16)
fn project(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = int2(textureDimensions(screen));
    let screen_size_f = float2(screen_size);

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= uint(screen_size.x) || id.y >= uint(screen_size.y)) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(id.y) + .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / screen_size_f;

    let col = textureSampleLevel(channel0, bilinear, uv, 0.).xyz;
    let pos = sqrt(col);

    let screenCoord = clamp(int2(toScreen(float4(pos,0.), screen_size_f)),int2(0,0),screen_size);

    let idx1 = int(screenCoord.x) + int(screen_size.x * screenCoord.y);

    atomicAdd(&atomic_storage[idx1*4+0], int(256. * col.x));
    atomicAdd(&atomic_storage[idx1*4+1], int(256. * col.y));
    atomicAdd(&atomic_storage[idx1*4+2], int(256. * col.z));
    atomicAdd(&atomic_storage[idx1*4+3], 1);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(id.y) + .5);

    let idx = int(id.x) + int(screen_size.x * id.y);

    let count = atomicLoad(&atomic_storage[idx*4+3]);
    let x = float(atomicLoad(&atomic_storage[idx*4+0]))/(float(count)*256.0);
    let y = float(atomicLoad(&atomic_storage[idx*4+1]))/(float(count)*256.0);
    let z = float(atomicLoad(&atomic_storage[idx*4+2]))/(float(count)*256.0);


    let projectCol = float3(x,y,z);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / float2(screen_size);

    let col = textureSampleLevel(channel0, bilinear, uv, 0.).xyz;

    let result = select(col, projectCol, count > 0);

    //let result = col;

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(result, 1.));
}
