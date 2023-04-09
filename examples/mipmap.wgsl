// WGSL compute shaders don't support automatic gradients or implicit LODs, so you need to do this manually

fn texCoord(fragCoord: float2) -> float2 {
    let screen_size = uint2(textureDimensions(screen));
    let uv = fragCoord / float2(screen_size);
    return float2(uv.x - .5, 1.) / -(1. - uv.y);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    let st = texCoord(fragCoord);
    let ddx = texCoord(fragCoord + float2(1.,0.)) - texCoord(fragCoord);
    let ddy = texCoord(fragCoord + float2(0.,1.)) - texCoord(fragCoord);
    let col = textureSampleGrad(channel0, trilinear, fract(st), ddx, ddy).rgb;

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
