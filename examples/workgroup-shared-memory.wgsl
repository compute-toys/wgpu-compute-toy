// An example of using shared memory to compute gradients across threads, allowing automatic LOD selection without having to duplicate computations

// Global variables
// private = current thread only
// workgroup = shared with all threads in workgroup
var<private> local_invocation_id_: uint3;
var<workgroup> texCoords: array<array<float2, 16>, 16>;

fn swap(x: uint) -> uint {
    // swap adjacent pairs of indices
    // 0 1 2 3 4 5
    // 1 0 3 2 5 4
    return select(x - 1u, x + 1u, x % 2u == 0u);
}

fn textureSampleMipmap(ch: texture_2d<f32>, coords: float2) -> float4 {
    let x = local_invocation_id_.x;
    let y = local_invocation_id_.y;

    // Share texture coordinates within the workgroup
    texCoords[x][y] = coords;

    // Synchronise threads in workgroup
    workgroupBarrier();

    // Read texture coordinates from adjacent threads in quad
    let ddx = texCoords[swap(x)][y] - coords;
    let ddy = texCoords[x][swap(y)] - coords;

    return textureSampleGrad(ch, trilinear_repeat, coords, ddx, ddy);
}

@compute @workgroup_size(16, 16)
fn main_image(
    @builtin(global_invocation_id) id: uint3,
    @builtin(local_invocation_id) local: uint3
) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / float2(screen_size);

    // Store global variable
    local_invocation_id_ = local;

    // Sample texture with automatically selected mip level
    let col = textureSampleMipmap(channel0, float2(uv.x - .5, 1.) / -(1. - uv.y) + .1 * time.elapsed).rgb;

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x < screen_size.x && id.y < screen_size.y) {
        // Output to screen (linear colour space)
        textureStore(screen, int2(id.xy), float4(col, 1.));
    }
}
