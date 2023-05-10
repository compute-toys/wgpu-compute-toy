#storage atomic_storage array<atomic<i32>>

const MaxSamples = 8.0;
const FOV = 0.8;
const PI = 3.14159265;
const TWO_PI = 6.28318530718;
const STEP = 0.01;
const LARGENUM = 1e10;

alias distanceArr = array<float,8>;

struct Camera
{
  pos: float3,
  cam: float3x3,
  fov: float,
  size: float2
}

struct TraceRes
{
    ro: float3,
    ex: float4,
    hit: bool,
    distances: distanceArr,
}

struct Ray
{
    ro: float3,
    rd: float3,
}

var<private> camera : Camera;
var<private> state : uint4;

fn getTresholds() -> distanceArr
{
    var arr: distanceArr;
    arr[0] = -log(0.9);
    arr[1] = -log(0.6);
    arr[2] = -log(0.5);
    arr[3] = -log(0.25);
    arr[4] = -log(0.15);
    arr[5] = -log(0.1);
    arr[6] = -log(0.05);
    arr[7] = -log(0.005);
    return arr;
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

fn udir(rng: float2) -> float3
{
    let r = float2(2.*PI*rng.x, acos(2.*rng.y - 1.0));
    let c = cos(r);
    let s = sin(r);
    return float3(c.x*s.y, s.x*s.y, c.y);
}

fn disk(rng: float2) -> float2
{
    return float2(sin(TWO_PI*rng.x), cos(TWO_PI*rng.x))*sqrt(rng.y);
}

fn Rotate(t: float) -> float2x2
{
    return float2x2(
        cos(t), sin(t),
      - sin(t), cos(t),
    );
}

fn RotXY(x: float3, t: float) -> float3
{
    return float3(Rotate(t)*x.xy, x.z);
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
    camera.pos = - (camera.cam*float3(5.0*custom.Radius+0.5,0.0,0.0));
    camera.size = screen_size_f;
}


//recover approximate transmittense from texture stored depths
fn ReTrace(td: float, distances: distanceArr) -> float
{
    let thresholds = getTresholds();

    if(td < distances[0])
    {
        return mix(0.0, thresholds[0], (td - 0.0)/(distances[0] - 0.0));
    }

    for(var i = 1; i < 8; i++)
    {
        if(td < distances[i])
        {
            let t = (td - distances[i - 1]) / (distances[i] - distances[i - 1]);
            return mix(thresholds[i - 1], thresholds[i], t);
        }
    }

    return thresholds[7];
}


fn sdMandelbulb(p: float3) -> float4
{
    var w = p;
    var m = dot(w,w);

    var trap = float4(abs(w),m);
	var dz = 1.0;

	for(var i = 0; i < 3; i++)
    {
        // trigonometric version (MUCH faster than polynomial)
		dz = 8.0*pow(m,3.5)*dz + 1.0;

        let r = length(w);
        let b = 8.0*acos( w.y/r);
        let a = 8.0*atan2( w.x, w.z );
        w = p + pow(r,8.0) * vec3( sin(b)*sin(a), cos(b), sin(b)*cos(a) );

        trap = min( trap, vec4(abs(w),m) );

        m = dot(w,w);
		if( m > 256.0 )
        {
            break;
        }
    }
    // distance estimation (through the Hubbard-Douady potential)
    return float4(trap.yzw, 0.25*log(m)*sqrt(m)/dz);
}

fn map(p: float3) -> float4
{
    return sdMandelbulb(p);
}

//trace up to a scattering threshold
fn Trace(ro: float3, rd: float3, threshold: float, maxDistance: float) -> TraceRes
{
    var c = float4(0.); //rgb extinction and w scattering
    let r = rand4();
    var td = STEP*r.x;

    var distances: distanceArr;
    for(var i = 0; i < 8; i++)
    {
        distances[i] = 100.0;
    }

    var j = 0;
    let thresholds = getTresholds();
    for(var i = 0; i < 150; i++)
    {
        let d = map(ro.xyz + rd*td); //distance
        var dx = mix(0.5, 1.0, r.y) * select(d.w+STEP, STEP - d.w, d.w < 0.0); //ray step

        let rho = select(float4(0.0,0.0,0.0,0.1), 40.0*custom.Scatter*float4(d.xyz, 1.0), d.w < 0.0);

        let dc = rho*float4(float3(10.0*custom.Absorption), 1.0);

        if(c.w + dc.w*dx > threshold)
        {
            dx = (threshold - c.w)/dc.w;
            td += dx;
            c += dc*dx;
            break;
        }

        if((c.w + dc.w*dx > thresholds[j]) && (j < 8))
        {
            distances[j] = td + (thresholds[j] - c.w)/dc.w;
            j++;
        }

        c += dc*dx;
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
    res.distances = distances;
    return res;
}

//project to clip space
fn Project(cam: Camera, p: float3) -> float3
{
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos)/td;
    let screen = dir*cam.cam;
    return float3(screen.yz*cam.size.y/(cam.fov*screen.x) + 0.5*cam.size,screen.x*td);
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
    let scaledColor = int3(256.0 * color/(depth*depth + 0.2));

    if(scaledColor.x>0)
    {
        atomicAdd(&atomic_storage[index*4+0], scaledColor.x);
    }

    if(scaledColor.y>0)
    {
        atomicAdd(&atomic_storage[index*4+1], scaledColor.y);
    }

    if(scaledColor.z>0)
    {
        atomicAdd(&atomic_storage[index*4+2], scaledColor.z);
    }
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

    var dist: distanceArr;
    let camPass1 = textureLoad(pass_in, screenCoord, 1, 0);
    for(var i = 0; i < 3; i++)
    {
        dist[i] = camPass1[i];
    }

    let camPass2 = textureLoad(pass_in, screenCoord, 2, 0);
    for(var i = 0; i < 3; i++)
    {
        dist[i+4] = camPass2[i];
    }


    let visibility =exp(-ReTrace(distance(camera.pos, pos),dist));

    let idx = screenCoord.x + screen_size.x * screenCoord.y;
    AdditiveBlend(visibility*color, projectedPos.z, idx);
}


fn getRay() -> Ray
{
    let r = rand4();

    var ray: Ray;

    if(r.x<custom.Laser)  //lazer
    {
       ray.ro = vec3(1.0,0.8,1.3)+r.yzw*0.2;
       ray.rd = vec3(-1.,-1.,-1.0)+r.yzw*0.00;
    }
    else
    {
       ray.ro = vec3(1.5, 2.0*custom.LightR*disk(r.yz));
       ray.rd = normalize(vec3(-50.0+0.02*r.w,0.,0.) - ray.ro);
    }

    ray.ro = RotXY(ray.ro, -PI*custom.LightAng);
    ray.rd = RotXY(ray.rd, -PI*custom.LightAng);

    return ray;
}

//the opposite of path tracing, the actual way light is propagating
fn ForwardTrace()
{
    //generate light rays at light sources
    var ray = getRay();
    var light = float4(1.3);

    let max_bounce = int(1.0 + 8.0*custom.Bounces);

    for(var i = 0; i < max_bounce; i++)
    {
        let r = rand4();
        let res = Trace(ray.ro, ray.rd, -log(r.x), 10.0);
        if(res.hit)
        {
            light *= res.ex;
            ray.ro = res.ro;
            ray.rd = udir(r.yz);

            //rasterize scattered light
            RasterizePoint(ray.ro, light.xyz);
        }
        else
        {
            return;
        }
    }
}

fn CameraRay(pix: float2) -> Ray
{
    let clip = (pix - 0.5*camera.size)/camera.size.y;
    var ray: Ray;
    ray.ro = camera.pos;
    ray.rd = camera.cam * normalize(float3(1.0, camera.fov*clip.xy));
    return ray;
}

@compute @workgroup_size(16, 16)
fn CameraPass(@builtin(global_invocation_id) id: uint3)
{
    let screen_size = uint2(textureDimensions(screen));
    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    SetCamera();

    var ray = CameraRay(float2(id.xy));

    let res = Trace(ray.ro, ray.rd, 8.0, 10.0);

    let td = res.distances;

    // Output to buffer
    textureStore(pass_out, int2(id.xy), 1, float4(td[0],td[1],td[2],td[3]));

    textureStore(pass_out, int2(id.xy), 2, float4(td[4],td[5],td[6],td[7]));
}


@compute @workgroup_size(16, 16)
fn Rasterize(@builtin(global_invocation_id) id: uint3)
{
    SetCamera();

    //RNG state
    state = uint4(id.x, id.y, id.z, uint(custom.NoiseAnimation)*time.frame);

    for(var i: i32 = 0; i < int(custom.Samples*MaxSamples + 1.0); i++)
    {
       ForwardTrace();
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

    color = tanh(2.8*float3(x,y,z)/(floor(custom.Samples*MaxSamples) + 1.0));

    return abs(color);
}

@compute @workgroup_size(16, 16)
fn FinalPass(@builtin(global_invocation_id) id: uint3)
{
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(id.y) + .5);


    let oldColor = textureLoad(pass_in, int2(id.xy), 0, 0);

    var color = float4(Sample(int2(id.xy)), 1.0)*1e-3;



    if(mouse.click != 1)
    {
       color += oldColor * custom.Accumulation;
    }

    // Output to buffer
    textureStore(pass_out, int2(id.xy), 0, color);

    // Output to screen
    //if(id.x >= screen_size.x/2u)
    {
        textureStore(screen, int2(id.xy), float4(color.xyz/color.w, 1.));
    }
    //else
    //{
    //    var camPass = textureLoad(pass_in, int2(id.xy), 1, 0);
    //    textureStore(screen, int2(id.xy), float4((camPass.xyz)/10.0, 1.));
   // }
}
