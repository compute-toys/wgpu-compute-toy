#include "Dave_Hoskins/hash"

#define noise_simplex_2d_hash hash22
#include "iq/noise_simplex_2d"

#define noise_simplex_3d_hash hash33
#include "nikat/noise_simplex_3d"

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }
    let fragCoord = float2(id.xy) + .5;
    let resolution = float2(screen_size);
    let p = fragCoord / resolution.xy;
    let p3 = float3(fragCoord / resolution.x, time.elapsed*0.025);

    var uv = p * float2(resolution.x / resolution.y, 1.) + time.elapsed * .25;
    var f = 0.;
    if (p.x < .6) { // left: value noise
        if (p.y < .5) {
            f = noise_simplex_2d( 16.0*uv );
        } else {
            f = noise_simplex_3d(p3*32.0);
        }
    } else { // right: fractal noise (4 octaves)
        if (p.y < .5) {
            uv *= 5.0;
            let m = mat2x2<f32>( 1.6,  1.2, -1.2,  1.6 );
            f  = 0.5000*noise_simplex_2d( uv ); uv = m*uv;
            f += 0.2500*noise_simplex_2d( uv ); uv = m*uv;
            f += 0.1250*noise_simplex_2d( uv ); uv = m*uv;
            f += 0.0625*noise_simplex_2d( uv ); uv = m*uv;
        } else {
            let m = p3*8.0+8.0;
            // directional artifacts can be reduced by rotating each octave
            let rot1 = mat3x3<f32>(-0.37, 0.36, 0.85,-0.14,-0.93, 0.34,0.92, 0.01,0.4);
            let rot2 = mat3x3<f32>(-0.55,-0.39, 0.74, 0.33,-0.91,-0.24,0.77, 0.12,0.63);
            let rot3 = mat3x3<f32>(-0.71, 0.52,-0.47,-0.08,-0.72,-0.68,-0.7,-0.45,0.56);
            f = 0.5333333*noise_simplex_3d(m*rot1)
              + 0.2666667*noise_simplex_3d(2.0*m*rot2)
              + 0.1333333*noise_simplex_3d(4.0*m*rot3)
              + 0.0666667*noise_simplex_3d(8.0*m);
        }
    }
    f = 0.5 + 0.5*f;
    f *= smoothstep( 0.0, 0.005, abs(p.x - 0.6) );

    f = pow(f, 2.2); // perceptual gradient to linear colour space
    textureStore(screen, int2(id.xy), float4(f, f, f, 1.));
}
