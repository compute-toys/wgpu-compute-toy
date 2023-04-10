// This allows you to dispatch an entrypoint multiple times. The uniform `dispatch.id` tells you which iteration is currently executing

#dispatch_count main_image 3

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    // Normalised pixel coordinates (from 0 to 1)
    let uv = fragCoord / float2(screen_size);

    // draw a colour band for each dispatch
    var col = float3(0.);
    if (uint(3. * uv.x) == dispatch.id) {
        col[dispatch.id] = 1.;
        textureStore(screen, int2(id.xy), float4(col, 1.));
    }
}
