// 2022 David A Roberts <https://davidar.io/>
// A simple example of randomly sampling locations from a HDR environment map according to lighting intensity

fn pcg_random(seed: ptr<function, uint>) -> float {
    *seed = *seed * 747796405u + 2891336453u;
    let word = ((*seed >> ((*seed >> 28u) + 4u)) ^ *seed) * 277803737u;
    return float((word >> 22u) ^ word) / float(0xffffffffu);
}

// weighted coin flip (bernoulli)
fn flip(state: ptr<function, uint>, p: float) -> bool {
    return pcg_random(state) <= p;
}

// relative weight of given region of image
fn weight(pos: int2, mip: int) -> float {
    return length(textureLoad(channel0, pos, mip).rgb);
}

// sample location from image according to pixel weights
fn sample_coord(state: ptr<function, uint>, mipmax: int) -> int2 {
    var pos = int2(0,0);
    for (var mip = mipmax - 1; mip >= 0; mip -= 1) {
        pos *= 2;
        let w00 = weight(pos + int2(0,0), mip);
        let w01 = weight(pos + int2(0,1), mip);
        let w10 = weight(pos + int2(1,0), mip);
        let w11 = weight(pos + int2(1,1), mip);
        let w0 = w00 + w01; // weight of column 0
        let w1 = w10 + w11; // weight of column 1
        let w = w0 + w1; // total weight
        pos += select(
            int2(0, select(0, 1, flip(state, w01 / w0))), // cond prob of row 1 given col 0
            int2(1, select(0, 1, flip(state, w11 / w1))), // cond prob of row 1 given col 1
            flip(state, w1 / w)); // prob of col 1
    }
    return pos;
}

@compute @workgroup_size(16, 16)
fn blank(@builtin(global_invocation_id) id: uint3) {
    textureStore(screen, int2(id.xy), float4(0.,0.,0.,1.));
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    var seed = id.x + id.y * screen_size.x + time.frame * screen_size.x * screen_size.y;
    let mipmax = int(textureNumLevels(channel0)) - 1;
    let uv = float2(sample_coord(&seed, mipmax)) / float2(textureDimensions(channel0));
    if (uv.y > 1.) { return; }
    if (flip(&seed, .25)) { // splat 25% of sampled points to screen
        textureStore(screen, int2(uv * float2(textureDimensions(screen))), float4(1.));
    }
}
