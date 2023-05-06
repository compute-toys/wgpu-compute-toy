#storage atomic_storage array<atomic<i32>>

const MaxSamples = 8.0;
const FOV = 0.8;
const PI = 3.14159265;
const TWO_PI = 6.28318530718;
//sqrt of particle count
const PARTICLE_COUNT = 2000;

const DEPTH_MIN = 0.2;
const DEPTH_MAX = 5.0;
const DEPTH_BITS = 16u;
const dq = float2(0.0, 1.0);
const eps = 0.01;

struct Camera
{
  pos: float3,
  cam: float3x3,
  fov: float,
  size: float2
}

const KerrM = 1.0;

struct GeodesicRay
{
    q:  float4,
    qt: float4,
    p:  float4,
};

struct Particle
{
    position: float4,
    velocity: float4,
}

var<private> camera : Camera;
var<private> state : uint4;
var<private> bokehRad : float;

fn sqr(x: float) -> float
{
    return x*x;
}

fn diag(a: float4) -> float4x4
{
    return float4x4(
        a.x,0.0,0.0,0.0,
        0.0,a.y,0.0,0.0,
        0.0,0.0,a.z,0.0,
        0.0,0.0,0.0,a.w
    );
}

fn pcg4d(a: uint4) -> uint4
{
	var v = a * 1664525u + 1013904223u;
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    v = v ^  ( v >> uint4(16u) );
    v.x += v.y*v.w; v.y += v.z*v.x; v.z += v.x*v.y; v.w += v.y*v.z;
    return v;
}

fn rand4() -> float4
{
    state = pcg4d(state);
    return float4(state)/float(0xffffffffu);
}

fn nrand4(sigma: float, mean: float4) -> float4
{
    let Z = rand4();
    return mean + sigma * sqrt(-2.0 * log(Z.xxyy)) *
           float4(cos(TWO_PI * Z.z),sin(TWO_PI * Z.z),cos(TWO_PI * Z.w),sin(TWO_PI * Z.w));
}

fn disk(r: float2) -> float2
{
    return vec2(sin(TWO_PI*r.x), cos(TWO_PI*r.x))*(r.y);
}

fn GetCameraMatrix(ang: float2) -> float3x3
{
    let x_dir = float3(cos(ang.x)*sin(ang.y), cos(ang.y), sin(ang.x)*sin(ang.y));
    let y_dir = normalize(cross(x_dir, float3(0.0,1.0,0.0)));
    let z_dir = normalize(cross(y_dir, x_dir));
    return float3x3(-x_dir, y_dir, z_dir);
}

fn SetCamera(ang: float2, fov: float)
{
    camera.fov = fov;
    camera.cam = GetCameraMatrix(ang);
    camera.pos = - (camera.cam*float3(50.0*custom.Radius+0.5,0.0,0.0));
    camera.size = float2(textureDimensions(screen));
}

//project to clip space
fn Project(cam: Camera, p: float3) -> float3
{
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos)/td;
    let screen = dir*cam.cam;
    return float3(screen.yz*cam.size.y/(cam.fov*screen.x) + 0.5*cam.size,screen.x*td);
}

fn LoadParticle(pix: int2) -> Particle
{
    var p: Particle;
    p.position = textureLoad(pass_in, pix, 0, 0);
    p.velocity = textureLoad(pass_in, pix, 1, 0);
    return p;
}

fn SaveParticle(pix: int2, p: Particle)
{
    textureStore(pass_out, pix, 0, p.position);
    textureStore(pass_out, pix, 1, p.velocity);
}

fn KerrGetR2(p: float3) -> float
{
    let rho = dot(p,p) - sqr(custom.KerrA);
    let r2 = 0.5*(rho + sqrt(sqr(rho) + sqr(2.0*custom.KerrA*p.z)));
    return r2;
}

fn KerrGetK(p: float3) -> float4
{
    let r2 = KerrGetR2(p);
    let r = sqrt(r2);
    let invr2 = 1.0 / (r2 + sqr(custom.KerrA) + 1e-3);
    let  k = float3((r*p.x - custom.KerrA*p.y) * invr2, (r*p.y + custom.KerrA*p.x) * invr2, p.z/(r + 1e-4));
    let f = r2 * (2.0 * KerrM * r - sqr(custom.KerrQ)) / (r2 * r2 + sqr(custom.KerrA * p.z) + 1e-3);
    return float4(k, f);
}

fn G(q: float4) -> float4x4
{
    //Kerr metric in Kerr-Schild coordinates
    let k = KerrGetK(q.yzw);
    let kf = k.w*float4(1.0, k.xyz);
    return diag(float4(-1.0,1.0,1.0,1.0)) + float4x4(kf, k.x*kf, k.y*kf, k.z*kf);
}

fn Ginv(q: float4) -> float4x4
{
    //inverse of Kerr metric in Kerr-Schild coordinates
    let k = KerrGetK(q.yzw);
    let kf = k.w*vec4(1.0, -k.xyz)/dot(k.xyz, k.xyz);
    return diag(float4(-1.0,1.0,1.0,1.0)) + float4x4(-kf, k.x*kf, k.y*kf, k.z*kf);
}

//lagrangian
fn Lmat(qt: float4, g: float4x4) -> float
{
    return   g[0][0]*qt.x*qt.x + g[1][1]*qt.y*qt.y + g[2][2]*qt.z*qt.z + g[3][3]*qt.w*qt.w +
        2.0*(g[0][1]*qt.x*qt.y + g[0][2]*qt.x*qt.z + g[0][3]*qt.x*qt.w +
                g[1][2]*qt.y*qt.z + g[1][3]*qt.y*qt.w +
                g[2][3]*qt.z*qt.w);
}

fn L(qt: float4, q: float4) -> float
{
    return Lmat(qt, G(q));
}

fn H(p: float4, ginv: float4x4) -> float
{
    return Lmat(p, ginv);
}

fn  ToMomentum(ray: GeodesicRay) -> float4
{
    return G(ray.q)*ray.qt;
}

fn  FromMomentum(ray: GeodesicRay) -> float4
{
    return Ginv(ray.q)*ray.p;
}

fn ParticleToGeodesic(particle: Particle) -> GeodesicRay
{
    var ray: GeodesicRay;
    ray.q = particle.position;
    ray.p = particle.velocity;
    return ray;
}

fn GeodesicToParticle(ray: GeodesicRay) -> Particle
{
    var particle: Particle;
    particle.position = ray.q;
    particle.velocity = ray.p/length(ray.qt);
    return particle;
}

fn HamiltonianGradient(ray: GeodesicRay) -> float4
{
    let ginv = Ginv(ray.q);
    let H0 = H(ray.p, ginv);
    let delta = 0.1;
    return (float4(
        L(ray.qt,ray.q+delta*dq.yxxx),
        L(ray.qt,ray.q+delta*dq.xyxx),
        L(ray.qt,ray.q+delta*dq.xxyx),
        L(ray.qt,ray.q+delta*dq.xxxy)) - H0)/delta;
}

fn VelClamp(vel: float4) -> float4
{
    return vel;//float4(vel.x, vel.yzw / max(1.0, length(vel.yzw)));
}

@compute @workgroup_size(16, 16)
fn SimulateParticles(@builtin(global_invocation_id) id: uint3)
{
    var pix = int2(id.xy);
    var p = LoadParticle(pix);

    if(pix.x > PARTICLE_COUNT || pix.y > PARTICLE_COUNT)
    {
        return;
    }

    state = uint4(id.x, id.y, id.z, time.frame);

    let r = sqrt(KerrGetR2(p.position.yzw));

    if(time.frame == 0u || r < 0.9 || r > 30.0)
    {
        let rng = rand4();
        let rng1 = rand4();
        p.position = 30.0*float4(1.0, 1.0, 1.0, custom.InitThick) * float4(0.0,2.0*rng.xyz - 1.0);

        let r01 = sqrt(KerrGetR2(p.position.yzw));
        if(r01 < 0.9)
        {
            return;
        }

        var vel = normalize(cross(p.position.yzw, float3(0.0,0.0,1.0)));

        vel += 0.3*(rng1.xyz * 0.5 - 0.25);
        let vscale = clamp(1.0 / (0.2 + 0.08*r01), 0., 1.0);
        p.velocity = float4(-1.0,2.0*(custom.InitSpeed - 0.5)*vel*vscale);
    }



    var ray = ParticleToGeodesic(p);

    if(mouse.click == 1)
    {
       // return;
    }

    for(var i = 0; i < int(custom.Steps*16.0 + 1.0); i++)
    {
        ray.qt = FromMomentum(ray);
        let qt0 = ray.qt;
        let dt = 0.5 * custom.TimeStep / (abs(ray.qt.x) + 0.01);
        ray.p += HamiltonianGradient(ray)*dt;
        ray.qt = FromMomentum(ray);
        ray.q += (ray.qt+qt0)*dt;
    }

    SaveParticle(pix, GeodesicToParticle(ray));
}

@compute @workgroup_size(16, 16)
fn Clear(@builtin(global_invocation_id) id: uint3) {
    let screen_size = int2(textureDimensions(screen));
    let idx0 = int(id.x) + int(screen_size.x * int(id.y));

    atomicStore(&atomic_storage[idx0*4+0], 0);
    atomicStore(&atomic_storage[idx0*4+1], 0);
    atomicStore(&atomic_storage[idx0*4+2], 0);
    atomicStore(&atomic_storage[idx0*4+3], 0);
}

fn AdditiveBlend(color: float3, depth: float, index: int)
{
    let scaledColor = 256.0 * color/depth;

    atomicAdd(&atomic_storage[index*4+0], int(scaledColor.x));
    atomicAdd(&atomic_storage[index*4+1], int(scaledColor.y));
    atomicAdd(&atomic_storage[index*4+2], int(scaledColor.z));
}


fn RasterizePoint(pos: float3, color: float3)
{
    let screen_size = int2(camera.size);
    let projectedPos = Project(camera, pos);

    let screenCoord = int2(projectedPos.xy);

    //outside of our view
    if(screenCoord.x < 0 || screenCoord.x >= screen_size.x ||
        screenCoord.y < 0 || screenCoord.y >= screen_size.y || projectedPos.z < 0.0)
    {
        return;
    }

    let idx = screenCoord.x + screen_size.x * screenCoord.y;

    AdditiveBlend(color, projectedPos.z, idx);
}

@compute @workgroup_size(16, 16)
fn Rasterize(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = int2(textureDimensions(screen));
    let screen_size_f = float2(screen_size);

    let ang = float2(mouse.pos.xy)*float2(-TWO_PI, PI)/screen_size_f + 1e-4;

    SetCamera(ang, FOV);

    //RNG state
    state = uint4(id.x, id.y, id.z, 0u);

    let rng = rand4();
    bokehRad = pow(rng.x, custom.BlurExponent1);

    if(mouse.click == 1 && custom.AnimatedNoise > 0.5)
    {
        state.w = time.frame;
    }

    var pix = int2(id.xy);

    if(pix.x > PARTICLE_COUNT || pix.y > PARTICLE_COUNT)
    {
        return;
    }

    var p = LoadParticle(pix);

    var pos = p.position.ywz;
    let vel = abs(p.velocity.ywz);
    var col = clamp(8.5*abs(vel)*dot(vel,vel)+0.1, float3(0.0), float3(5.0));
    col /= (0.1+bokehRad);
    let impSample = (col.x + col.y + col.z)*bokehRad;
    let sampleCount = clamp(int(impSample*custom.Samples*MaxSamples + 1.0), 1, 1024);
    let normalCount = int(custom.Samples*MaxSamples + 1.0);

    col *= float(normalCount)/float(sampleCount);

    for(var i = 0; i < sampleCount; i++)
    {
        let R = 2.0*custom.BlurRadius*bokehRad;
        let rng = rand4();
        let dpos = R*normalize(nrand4(1.0, float4(0.0)).xyz)*pow(rng.x, custom.BlurExponent2);
        RasterizePoint(pos + dpos, col);
    }
}

fn Sample(pos: int2) -> float3
{
    let screen_size = int2(textureDimensions(screen));
    let idx = pos.x + screen_size.x * pos.y;

    var color: float3;
        let x = float(atomicLoad(&atomic_storage[idx*4+0]))/(256.0);
        let y = float(atomicLoad(&atomic_storage[idx*4+1]))/(256.0);
        let z = float(atomicLoad(&atomic_storage[idx*4+2]))/(256.0);

        color = tanh(custom.Exposure*0.03*float(screen_size.x)*float3(x,y,z)/(custom.Samples*MaxSamples + 1.0));

    return abs(color);
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3)
{
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(id.y) + .5);


    var color = float4(Sample(int2(id.xy)),1.0);

    let oldColor = textureLoad(pass_in, int2(id.xy), 2, 0);

    if(mouse.click == 1 && custom.AnimatedNoise > 0.5)
    {
        color += oldColor * custom.Accumulation;
    }

    // Output to buffer
    textureStore(pass_out, int2(id.xy), 2, color);

    textureStore(screen, int2(id.xy), float4(color.xyz/color.w, 1.));
}
