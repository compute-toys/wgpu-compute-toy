#include <float>

#include "Dave_Hoskins/hash"

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

    // Time varying pixel colour
    var col = .5 + .5 * cos(time.elapsed + uv.xyxx + float4(0.,2.,4.,0.));

    col.x /= floor(3. * hash12(fragCoord));
    col.z -= hash12(fragCoord);
    col.z = log(col.z);
    if (time.frame % 199u == 0u) { col.w = col.x; }

    assert_toy(0, !isinf(col.x));
    assert_toy(1, isfinite(col.y));
    assert_toy(2, !isnan(col.z));
    assert_toy(3, isnormal(col.w));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), col);
}
