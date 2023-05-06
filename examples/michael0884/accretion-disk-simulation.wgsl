// Accretion disk around a Kerr-Newman black hole. A rather simple general relativistic hydrodynamics simulation without EM effects.

#storage atomic_storage array<atomic<i32>>

struct Camera
{
  pos: float3,
  cam: float3x3,
  fov: float,
  size: float2
}

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

struct TraceRes
{
    ro: float3,
    ex: float4,
    color: float3,
    hit: bool,
}

struct Ray
{
    ro: float3,
    rd: float3,
}

fn isfinite(x: f32) -> bool {
    return clamp(x, -3.4e38, 3.4e38) == x;
}

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
const KerrM = 1.0;
const ENCODE_SCALE = 64000.0;
const GridScale = float3(38.0, 38.0, 38.0);

var<private> GridSize : int3;
var<private> camera : Camera;
var<private> state : uint4;
var<private> bokehRad : float;
var<private> screen_size: int2;

fn Encode(x: float) -> int
{
    return int(x*ENCODE_SCALE);
}

fn Decode(x: int) -> float
{
    return float(x)/ENCODE_SCALE;
}

fn ComputeGridSize()
{
    let screenSize = float2(textureDimensions(screen));
    let maxVoxelCount = screenSize.x*screenSize.y;
    let sideSize = pow(maxVoxelCount/(GridScale.x*GridScale.y*GridScale.z), 0.333333);
    GridSize = int3(sideSize * GridScale);
}

fn ToGrid(pos: float3) -> float3
{
    return float3(GridSize)*(pos/GridScale + 0.5);
}

fn VoxelID(pos: int3) -> int
{
    return pos.x + (pos.y + pos.z*GridSize.y)*GridSize.x;
}

fn IndexToPixel(id: int) -> int2
{
    return int2(id%screen_size.x,id/screen_size.x);
}

fn Voxel(id: int) -> int3
{
    var pos: int3;
    pos.x = id%GridSize.x;
    pos.y = (id/GridSize.x)%GridSize.y;
    pos.z = id/(GridSize.x*GridSize.y);
    return pos;
}

fn sqr(x: float) -> float
{
    return x*x;
}

fn Kernel(dnode: float3) -> float
{
    //box overlaps
    let aabb0 = max(dnode - 0.5, float3(-0.5));
    let aabb1 = min(dnode + 0.5, float3(0.5));
    let size = max(aabb1 - aabb0, float3(0.0));
    return size.x*size.y*size.z;
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

fn SetCamera()
{
    let screen_size = int2(textureDimensions(screen));
    let screen_size_f = float2(screen_size);
    let ang = float2(mouse.pos.xy)*float2(-TWO_PI, PI)/screen_size_f + float2(0.4, 0.4);

    camera.fov = FOV;
    camera.cam = GetCameraMatrix(ang);
    camera.pos = - (camera.cam*float3(1.5*GridScale.x*custom.Radius+0.5,0.0,0.0));
    camera.size = screen_size_f;
}

//project to clip space
fn Project(cam: Camera, p: float3) -> float3
{
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos)/td;
    let screen = dir*cam.cam;
    return float3(screen.yz*cam.size.y/(cam.fov*screen.x) + 0.5*cam.size,screen.x*td);
}

fn CameraRay(pix: float2) -> Ray
{
    let clip = (pix - 0.5*camera.size)/camera.size.y;
    var ray: Ray;
    ray.ro = camera.pos;
    ray.rd = camera.cam * normalize(float3(1.0, camera.fov*clip.xy));
    return ray;
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
    let vel = particle.velocity.xyz;
    ray.p = float4(-sqrt(abs(1.0 - dot(vel,vel))), vel);
    return ray;
}

fn GeodesicToParticle(ray: GeodesicRay) -> Particle
{
    var particle: Particle;
    particle.position = ray.q;
    let normalized = normalize(ray.p);
    particle.velocity = float4(normalized.yzw, 0.0);
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
fn ClearGrid(@builtin(global_invocation_id) id: uint3)
{
    screen_size = int2(textureDimensions(screen));
    let idx0 = int(id.x) + int(screen_size.x * int(id.y));

    for(var i = 0; i<4; i++)
    {
        atomicStore(&atomic_storage[idx0*4+i], 0);
    }
}

fn AddToGrid(value: float, index: int)
{
    atomicAdd(&atomic_storage[index], Encode(value));
}

fn Scatter(data: float4, gridPos: float3, delta: int3, pressure: float)
{
    let voxel = int3(floor(gridPos)) + delta;

    if(voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
       voxel.x >= GridSize.x || voxel.y >= GridSize.y || voxel.z >= GridSize.z)
    {
        return;
    }

    let dpos = gridPos - float3(voxel);
    let weight = Kernel(dpos);

    let index = VoxelID(voxel);
    let scatterData = (data + float4(dpos*pressure, 0.0))*weight;

    AddToGrid(scatterData.x, 4*index + 0);
    AddToGrid(scatterData.y, 4*index + 1);
    AddToGrid(scatterData.z, 4*index + 2);
    AddToGrid(scatterData.w, 4*index + 3);
}

@compute @workgroup_size(16, 16)
fn P2G(@builtin(global_invocation_id) id: uint3)
{
    screen_size = int2(textureDimensions(screen));
    var pix = int2(id.xy);
    if(pix.x > PARTICLE_COUNT || pix.y > PARTICLE_COUNT ||
       pix.x  >= screen_size.x || pix.y  >= screen_size.y)
    {
        return;
    }

    var p = LoadParticle(pix);

    state = uint4(id.x, id.y, id.z, time.frame);

    let r = sqrt(KerrGetR2(p.position.yzw));
    if(time.frame == 0u || r < 1.05 || any(abs(p.position.yzw) > GridScale.xyz*0.5 - 1.0))
    {
        let rng = rand4();
        let rng1 = rand4();
        p.position = float4(1.0, GridScale) * float4(0.0,rng.xyz - 0.5);

        var vel = normalize(cross(p.position.yzw, float3(0.0,1.,1.0)));
        let r01 = sqrt(KerrGetR2(p.position.yzw));

        vel += 0.3*(rng1.xyz * 0.5 - 0.25);
        let vscale = clamp(1.0 / (0.2 + 0.08*r01), 0., 1.0);
        p.velocity = float4(2.0*(custom.InitSpeed - 0.5)*vel*vscale, 0.0);

        if(r01 < 1.05 || any(abs(p.position.yzw) > GridScale.xyz*0.5 - 1.0))
        {
            SaveParticle(pix, p);
            return;
        }
    }

    let density = p.velocity.w;
    var ray = ParticleToGeodesic(p);

    //assert(0, isfinite(ray.q.y));
    //assert(1, isfinite(ray.p.y));

    if(mouse.click == 1)
    {
       // return;
    }

    let steps = custom.Steps*64.0 + 1.0;
    for(var i = 0; i < int(steps); i++)
    {
        ray.qt = FromMomentum(ray);
        let qt0 = ray.qt;
        let dt = 0.5 * custom.TimeStep / (abs(ray.qt.x) + 0.01);
        ray.p += HamiltonianGradient(ray)*dt;
        ray.qt = FromMomentum(ray);
        ray.q += (ray.qt+qt0)*dt;
    }
    p = GeodesicToParticle(ray);
    SaveParticle(pix, p);

    //P2G
    ComputeGridSize();

    let gridPos = ToGrid(p.position.yzw);
    let data = float4(p.velocity.xyz, 1.0);
    let pressure = -steps*custom.TimeStep*custom.Pressure*density;
    for(var i = 0; i <= 1; i++)
    {
        for(var j = 0; j <= 1; j++)
        {
            for(var k = 0; k <= 1; k++)
            {
                Scatter(data, gridPos, int3(i,j,k), pressure);
            }
        }
    }
}


fn SampleGrid(pos: int3) -> float4
{
    var res: float4;
    let index = VoxelID(pos);
    return textureLoad(pass_in, IndexToPixel(index), 3, 0);
}

fn Trilinear(p: float3) -> float4
{
    let pos = ToGrid(p);
    let pi = int3(floor(pos));
    let pf = fract(pos);

    let a000 = SampleGrid(pi + int3(0,0,0));
    let a001 = SampleGrid(pi + int3(0,0,1));
    let a010 = SampleGrid(pi + int3(0,1,0));
    let a011 = SampleGrid(pi + int3(0,1,1));
    let a100 = SampleGrid(pi + int3(1,0,0));
    let a101 = SampleGrid(pi + int3(1,0,1));
    let a110 = SampleGrid(pi + int3(1,1,0));
    let a111 = SampleGrid(pi + int3(1,1,1));

    let a00 = mix(a000, a001, pf.z);
    let a01 = mix(a010, a011, pf.z);
    let a10 = mix(a100, a101, pf.z);
    let a11 = mix(a110, a111, pf.z);

    let a0 = mix(a00, a01, pf.y);
    let a1 = mix(a10, a11, pf.y);

    return mix(a0, a1, pf.x);
}

fn map(p: float3) -> float4
{
    return Trilinear(p);
}

@compute @workgroup_size(16, 16)
fn GridToTexture(@builtin(global_invocation_id) id: uint3)
{
    screen_size = int2(textureDimensions(screen));
    var pix = int2(id.xy);
    if(pix.x >= screen_size.x || pix.y >= screen_size.y)
    {
        return;
    }
    var res: float4;
    let index = int(id.x) + int(screen_size.x * int(id.y));

    res.x = Decode(atomicLoad(&atomic_storage[index*4+0]));
    res.y = Decode(atomicLoad(&atomic_storage[index*4+1]));
    res.z = Decode(atomicLoad(&atomic_storage[index*4+2]));
    res.w = Decode(atomicLoad(&atomic_storage[index*4+3]));

    textureStore(pass_out, int2(id.xy), 3, res);
}

fn Gather(gridPos: float3, delta: int3) -> float4
{
    let voxel = int3(floor(gridPos)) + delta;

    if(voxel.x < 0 || voxel.y < 0 || voxel.z < 0 ||
       voxel.x >= GridSize.x || voxel.y >= GridSize.y || voxel.z >= GridSize.z)
    {
        return float4(0.0);
    }

    let weight = Kernel(gridPos - float3(voxel));

    let index = VoxelID(voxel);

    var res: float4;
    res.x = Decode(atomicLoad(&atomic_storage[index*4+0]));
    res.y = Decode(atomicLoad(&atomic_storage[index*4+1]));
    res.z = Decode(atomicLoad(&atomic_storage[index*4+2]));
    res.w = Decode(atomicLoad(&atomic_storage[index*4+3]));

    return weight*float4(res.xyz / (res.w + 1e-3), res.w);
}


@compute @workgroup_size(16, 16)
fn G2P(@builtin(global_invocation_id) id: uint3)
{
    var pix = int2(id.xy);
    var p = LoadParticle(pix);

    screen_size = int2(textureDimensions(screen));
    if(pix.x > PARTICLE_COUNT || pix.y > PARTICLE_COUNT ||
       pix.x  >= screen_size.x || pix.y  >= screen_size.y)
    {
        return;
    }


    state = uint4(id.x, id.y, id.z, time.frame);

    ComputeGridSize();
    let pos = p.position.yzw;
    let gridPos = ToGrid(pos);

    var data = float4(0.000);
    for(var i = 0; i <= 1; i++)
    {
        for(var j = 0; j <= 1; j++)
        {
            for(var k = 0; k <= 1; k++)
            {
                data += Gather(gridPos, int3(i,j,k));
            }
        }
    }
    let cell = GridScale/float3(GridSize);
    let ddx = 0.25*float3(-1.0, 0.0, 1.0);
    let gradient = float3(
        map(pos + cell*ddx.zyy).w - map(pos + cell*ddx.xyy).w,
        map(pos + cell*ddx.yzy).w - map(pos + cell*ddx.yxy).w,
        map(pos + cell*ddx.yyz).w - map(pos + cell*ddx.yyx).w
    )/0.25;

    p.velocity = float4(mix(p.velocity.xyz, data.xyz, custom.FLIP), data.w);

    SaveParticle(pix, p);
}

fn Sample(pos: int2) -> float3
{
    let screen_size = int2(textureDimensions(screen));
    let idx = pos.x + screen_size.x * pos.y;

    var color: float3;
    let x = float(atomicLoad(&atomic_storage[idx*4+0]))/(256.0);
    let y = float(atomicLoad(&atomic_storage[idx*4+1]))/(256.0);
    let z = float(atomicLoad(&atomic_storage[idx*4+2]))/(256.0);

    color = tanh(custom.Exposure*0.03*float(screen_size.x)*abs(float3(x,y,z))/(custom.Samples*MaxSamples + 1.0));

    return abs(color);
}

//trace up to a scattering threshold
fn Trace(ro: float3, rd: float3, threshold: float, maxDistance: float) -> TraceRes
{
    var c = float4(0.); //rgb extinction and w scattering

    var dx = 2.0*GridScale.x/float(GridSize.x);
    var td = dx*rand4().x;
    var color = float3(0.0);
    for(var i = 0; i < 256; i++)
    {
        let cpos = ro.xyz + rd*td;

        if(any(abs(cpos) > GridScale.xzy*0.49 - 1.0))
        {
            td += dx;
            continue;
        }

        let d = map(cpos.xzy); //distance


        let rho = d.w;
        let dc = rho*float4(0.3,0.5,1.0,1.0)*0.1;

        if(c.w + dc.w*dx > threshold)
        {
            dx = (threshold - c.w)/dc.w;
            td += dx;
            c += dc*dx;
            break;
        }

        c += dc*dx;
        color += exp(-c.xyz) * dx * length(d.xyz) *rho * 0.4;
        td += dx;

        if(td > maxDistance)
        {
            break;
        }
    }

    var res: TraceRes;
    res.ro = ro + rd*td;
    res.ex = exp(-c);
    res.hit = c.w >= threshold;
    res.color = color;
    return res;
}

@compute @workgroup_size(16, 16)
fn Render(@builtin(global_invocation_id) id: uint3)
{
    screen_size = int2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (int(id.x) >= screen_size.x || int(id.y) >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(id.y) + .5);

    SetCamera();
    ComputeGridSize();

    state = uint4(id.x, id.y, id.z, time.frame);
    var ray = CameraRay(float2(id.xy));

    let res = Trace(ray.ro, ray.rd, 8.0, 2.0*GridScale.x);

    //var color = float4(Sample(int2(id.xy)),1.0);

    var color = float4(custom.Exposure*res.color, 1.0);

    let oldColor = textureLoad(pass_in, int2(id.xy), 2, 0);

    if(mouse.click == 1 && custom.AnimatedNoise > 0.5)
    {
        color += oldColor * custom.Accumulation;
    }

    // Output to buffer
    textureStore(pass_out, int2(id.xy), 2, color);

    textureStore(screen, int2(id.xy), float4(color.xyz/color.w, 1.));
}
