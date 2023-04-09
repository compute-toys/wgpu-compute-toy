// Based on the path tracing tutorial series by demofox:
// https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
// Ported to WGSL by davidar.io

const MINIMUM_RAY_HIT_TIME = .1;
const FAR_PLANE = 1e4;
const FOV_DEGREES = 90.;
const NUM_BOUNCES = 8;
const RAY_POS_NORMAL_NUDGE = .01;
const NUM_RENDERS_PER_FRAME = 100;
const EXPOSURE = .5;
const PI = 3.1415926536;

struct Material {
    albedo: float3,
    emissive: float3,
    specular: float3,
    percentSpecular: float,
    roughness: float,
    IOR: float,
}

struct RayHitInfo {
    dist: float,
    normal: float3,
    material: Material,
};

fn WangHash(seed: ptr<function, uint>) -> uint
{
    *seed = (*seed ^ 61u) ^ (*seed >> 16u);
    *seed *= 9u;
    *seed = *seed ^ (*seed >> 4u);
    *seed *= 0x27d4eb2du;
    *seed = *seed ^ (*seed >> 15u);
    return *seed;
}

fn RandomFloat01(seed: ptr<function, uint>) -> float
{
    return float(WangHash(seed)) / 4294967296.0;
}

fn RandomUnitVector(seed: ptr<function, uint>) -> float3
{
    let z = RandomFloat01(seed) * 2. - 1.;
    let a = RandomFloat01(seed) * 2. * PI;
    let r = sqrt(1. - z * z);
    let x = r * cos(a);
    let y = r * sin(a);
    return float3(x, y, z);
}

// ACES tone mapping curve fit to go from HDR to LDR
// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn ACESFilm(x: float3) -> float3
{
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x*(a*x + b)) / (x*(c*x + d) + e), float3(0.), float3(1.));
}

fn FresnelReflectAmount(
    n1: float,
    n2: float,
    normal: float3,
    incident: float3,
    f0: float,
    f90: float)
    -> float
{
    // Schlick aproximation
    var r0 = (n1-n2) / (n1+n2);
    r0 *= r0;
    var cosX = -dot(normal, incident);
    if (n1 > n2)
    {
        let n = n1/n2;
        let sinT2 = n*n*(1.0-cosX*cosX);
        // Total internal reflection
        if (sinT2 > 1.0)
        {
            return f90;
        }
        cosX = sqrt(1.0-sinT2);
    }
    let x = 1.0-cosX;
    let ret = r0+(1.0-r0)*x*x*x*x*x;

    // adjust reflect multiplier for object reflectivity
    return mix(f0, f90, ret);
}

fn ScalarTriple(u: float3, v: float3, w: float3) -> float
{
    return dot(cross(u, v), w);
}

fn TestQuadTrace(
    rayPos: float3,
    rayDir: float3,
    info: ptr<function, RayHitInfo>,
    _a: float3,
    _b: float3,
    _c: float3,
    _d: float3)
    -> bool
{
    var a = _a;
    var b = _b;
    var c = _c;
    var d = _d;
    // calculate normal and flip vertices order if needed
    var normal = normalize(cross(c-a, c-b));
    if (dot(normal, rayDir) > 0.)
    {
        normal *= -1.;

        var temp = d;
        d = a;
        a = temp;

        temp = b;
        b = c;
        c = temp;
    }

    let p = rayPos;
    let q = rayPos + rayDir;
    let pq = q - p;
    let pa = a - p;
    let pb = b - p;
    let pc = c - p;

    // determine which triangle to test against by testing against diagonal first
    let m = cross(pc, pq);
    var v = dot(pa, m);
    var intersectPos = float3(0.);
    if (v >= 0.)
    {
        // test against triangle a,b,c
        var u = -dot(pb, m);
        if (u < 0.) { return false; }
        var w = ScalarTriple(pq, pb, pa);
        if (w < 0.) { return false; }
        let denom = 1. / (u+v+w);
        u*=denom;
        v*=denom;
        w*=denom;
        intersectPos = u*a+v*b+w*c;
    }
    else
    {
        let pd = d - p;
        var u = dot(pd, m);
        if (u < 0.) { return false; }
        var w = ScalarTriple(pq, pa, pd);
        if (w < 0.) { return false; }
        v = -v;
        let denom = 1. / (u+v+w);
        u*=denom;
        v*=denom;
        w*=denom;
        intersectPos = u*a+v*d+w*c;
    }

    var dist = 0.;
    if (abs(rayDir.x) > 0.)
    {
        dist = (intersectPos.x - rayPos.x) / rayDir.x;
    }
    else if (abs(rayDir.y) > 0.)
    {
        dist = (intersectPos.y - rayPos.y) / rayDir.y;
    }
    else
    {
        dist = (intersectPos.z - rayPos.z) / rayDir.z;
    }

    if (dist > MINIMUM_RAY_HIT_TIME && dist < (*info).dist)
    {
        (*info).dist = dist;
        (*info).normal = normal;
        return true;
    }

    return false;
}

fn TestSphereTrace(
    rayPos: float3,
    rayDir: float3,
    info: ptr<function, RayHitInfo>,
    sphere: float4)
    -> bool
{
    //get the vector from the center of this sphere to where the ray begins.
    let m = rayPos - sphere.xyz;

    //get the dot product of the above vector and the ray's vector
    let b = dot(m, rayDir);

    let c = dot(m, m) - sphere.w * sphere.w;

    //exit if r's origin outside s (c > 0) and r pointing away from s (b > 0)
    if(c > 0.0 && b > 0.0)
    {
        return false;
    }

    //calculate discriminant
    let discr = b * b - c;

    //a negative discriminant corresponds to ray missing sphere
    if(discr < 0.0)
    {
        return false;
    }

    //ray now found to intersect sphere, compute smallest t value of intersection
    var fromInside = false;
    var dist = -b - sqrt(discr);
    if (dist < 0.)
    {
        fromInside = true;
        dist = -b + sqrt(discr);
    }

    if (dist > MINIMUM_RAY_HIT_TIME && dist < (*info).dist)
    {
        (*info).dist = dist;
        (*info).normal = normalize((rayPos+rayDir*dist) - sphere.xyz) * select(1., -1., fromInside);
        return true;
    }

    return false;
}

fn TestSceneTrace(rayPos: float3, rayDir: float3, hitInfo: ptr<function, RayHitInfo>)
{
    // to move the scene around, since we can't move the camera yet
    let sceneTranslation = float3(0., 0., 10.);
    let sceneTranslation4 = float4(sceneTranslation, 0.);

    // back wall
    {
        let A = float3(-12.6, -12.6, 25.) + sceneTranslation;
        let B = float3( 12.6, -12.6, 25.) + sceneTranslation;
        let C = float3( 12.6,  12.6, 25.) + sceneTranslation;
        let D = float3(-12.6,  12.6, 25.) + sceneTranslation;
        if (TestQuadTrace(rayPos, rayDir, hitInfo, A, B, C, D))
        {
            (*hitInfo).material.albedo = float3(.7);
            (*hitInfo).material.emissive = float3(0.);
        }
    }

    // floor
    {
        let A = float3(-12.6, -12.45, 25.) + sceneTranslation;
        let B = float3( 12.6, -12.45, 25.) + sceneTranslation;
        let C = float3( 12.6, -12.45, 15.) + sceneTranslation;
        let D = float3(-12.6, -12.45, 15.) + sceneTranslation;
        if (TestQuadTrace(rayPos, rayDir, hitInfo, A, B, C, D))
        {
            (*hitInfo).material.albedo = float3(.7);
            (*hitInfo).material.emissive = float3(0.);
        }
    }

    // ceiling
    {
        let A = float3(-12.6, 12.5, 25.0) + sceneTranslation;
        let B = float3( 12.6, 12.5, 25.0) + sceneTranslation;
        let C = float3( 12.6, 12.5, 15.0) + sceneTranslation;
        let D = float3(-12.6, 12.5, 15.0) + sceneTranslation;
        if (TestQuadTrace(rayPos, rayDir, hitInfo, A, B, C, D))
        {
            (*hitInfo).material.albedo = float3(.7);
            (*hitInfo).material.emissive = float3(0.);
        }
    }

    // left wall
    {
        let A = float3(-12.5, -12.6, 25.0) + sceneTranslation;
        let B = float3(-12.5, -12.6, 15.0) + sceneTranslation;
        let C = float3(-12.5,  12.6, 15.0) + sceneTranslation;
        let D = float3(-12.5,  12.6, 25.0) + sceneTranslation;
        if (TestQuadTrace(rayPos, rayDir, hitInfo, A, B, C, D))
        {
            (*hitInfo).material.albedo = float3(.5, 0., 0.);
            (*hitInfo).material.emissive = float3(0.);
        }
    }

    // right wall
    {
        let A = float3( 12.5, -12.6, 25.) + sceneTranslation;
        let B = float3( 12.5, -12.6, 15.) + sceneTranslation;
        let C = float3( 12.5,  12.6, 15.) + sceneTranslation;
        let D = float3( 12.5,  12.6, 25.) + sceneTranslation;
        if (TestQuadTrace(rayPos, rayDir, hitInfo, A, B, C, D))
        {
            (*hitInfo).material.albedo = float3(0., .5, 0.);
            (*hitInfo).material.emissive = float3(0.);
        }
    }

    // light
    {
        let A = float3(-5.0, 12.4,  22.5) + sceneTranslation;
        let B = float3( 5.0, 12.4,  22.5) + sceneTranslation;
        let C = float3( 5.0, 12.4,  17.5) + sceneTranslation;
        let D = float3(-5.0, 12.4,  17.5) + sceneTranslation;
        if (TestQuadTrace(rayPos, rayDir, hitInfo, A, B, C, D))
        {
            (*hitInfo).material.albedo = float3(0.);
            (*hitInfo).material.emissive = float3(1., .8, .5) * 20.;
        }
    }

    if (TestSphereTrace(rayPos, rayDir, hitInfo, vec4(-9., -9., 20., 3.)+sceneTranslation4))
    {
        (*hitInfo).material.albedo = float3(.9, .9, .5);
        (*hitInfo).material.emissive = float3(0.);
        (*hitInfo).material.specular = float3(.9);
        (*hitInfo).material.percentSpecular = .1;
        (*hitInfo).material.roughness = .2;
        (*hitInfo).material.IOR = 1.;
    }

    if (TestSphereTrace(rayPos, rayDir, hitInfo, vec4(0., -9., 20., 3.)+sceneTranslation4))
    {
        (*hitInfo).material.albedo = float3(.9, .5, .9);
        (*hitInfo).material.emissive = float3(0.);
        (*hitInfo).material.specular = float3(.9);
        (*hitInfo).material.percentSpecular = .3;
        (*hitInfo).material.roughness = .2;
        (*hitInfo).material.IOR = 1.;
    }

    if (TestSphereTrace(rayPos, rayDir, hitInfo, vec4(9., -9., 20., 3.)+sceneTranslation4))
    {
        (*hitInfo).material.albedo = float3(0., 0., 1.);
        (*hitInfo).material.emissive = float3(0.);
        (*hitInfo).material.specular = float3(1., 0., 0.);
        (*hitInfo).material.percentSpecular = .5;
        (*hitInfo).material.roughness = .4;
        (*hitInfo).material.IOR = 1.;
    }

    for(var i = 0; i < 5; i += 1) {
        if (TestSphereTrace(rayPos, rayDir, hitInfo, vec4(float(5*i - 10), 0., 23., 1.75)+sceneTranslation4))
        {
            (*hitInfo).material.albedo = float3(1.);
            (*hitInfo).material.emissive = float3(0.);
            (*hitInfo).material.specular = float3(.3, 1., .3);
            (*hitInfo).material.percentSpecular = 1.;
            (*hitInfo).material.roughness = float(i*i) / 16.;
            (*hitInfo).material.IOR = 1.;
        }
    }
}

fn GetColorForRay(startRayPos: float3, startRayDir: float3, rngState: ptr<function, uint>) -> float3
{

    var ret = float3(0.);
    var throughput = float3(1.);
    var rayPos = startRayPos;
    var rayDir = startRayDir;

    for (var bounceIndex = 0; bounceIndex <= NUM_BOUNCES; bounceIndex += 1)
    {
        var hitInfo = RayHitInfo();
        hitInfo.dist = FAR_PLANE;
        TestSceneTrace(rayPos, rayDir, &hitInfo);

        if (hitInfo.dist == FAR_PLANE)
        {
            let uv = float2(atan2(rayDir.x, rayDir.z) / (2.*PI), acos(rayDir.y) / PI);
            ret += textureSampleLevel(channel0, bilinear, fract(uv), 0.).rgb * throughput;
            break;
        }

        rayPos = rayPos + rayDir * hitInfo.dist + hitInfo.normal * RAY_POS_NORMAL_NUDGE;

        var specularChance = hitInfo.material.percentSpecular;
        if (specularChance > 0.)
        {
            specularChance = FresnelReflectAmount(
                1., hitInfo.material.IOR, rayDir, hitInfo.normal,
                hitInfo.material.percentSpecular, 1.);
        }
        let doSpecular = (RandomFloat01(rngState) < specularChance);

        let rayProbability = select(1. - specularChance, specularChance, doSpecular);

        let diffuseRayDir = normalize(hitInfo.normal + RandomUnitVector(rngState));
        var specularRayDir = reflect(rayDir, hitInfo.normal);
        specularRayDir = normalize(mix(specularRayDir, diffuseRayDir, hitInfo.material.roughness));
        rayDir = select(diffuseRayDir, specularRayDir, doSpecular);

        ret += hitInfo.material.emissive * throughput;

        throughput *= select(hitInfo.material.albedo, hitInfo.material.specular, doSpecular);

        throughput /= max(1e-3, rayProbability);

        let p = max(throughput.r, max(throughput.g, throughput.b));
        if (RandomFloat01(rngState) > p) {
            break;
        }
        throughput /= max(1e-3, p);
    }

    return ret;
}

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

    // initialize a random number state based on frag coord and frame
    var rngState = (id.x * 1973u + id.y * 9277u + time.frame * 26699u) | 1u;

    // The ray starts at the camera position (the origin)
    let rayPosition = float3(0.);

    // calculate the camera distance
    let cameraDistance = 1. / tan(FOV_DEGREES * .5 * PI / 180.);

    // calculate subpixel camera jitter for anti aliasing
    let jitter = float2(RandomFloat01(&rngState), RandomFloat01(&rngState)) - .5;

    // calculate coordinates of the ray target on the imaginary pixel plane.
    // -1 to +1 on x,y axis. 1 unit away on the z axis
    var rayTarget = float3((fragCoord + jitter) / float2(screen_size) * 2. - 1., cameraDistance);

    // correct for aspect ratio
    let aspectRatio = float(screen_size.x) / float(screen_size.y);
    rayTarget.y /= aspectRatio;

    // calculate a normalized vector for the ray direction.
    // it's pointing from the ray position to the ray target.
    let rayDir = normalize(rayTarget - rayPosition);

    // raytrace for this pixel
    var col = float3(0.);
    for (var index = 0; index < NUM_RENDERS_PER_FRAME; index += 1)
    {
        col += GetColorForRay(rayPosition, rayDir, &rngState) / float(NUM_RENDERS_PER_FRAME);
    }

    // average the frames together
    let lastFrameCol = textureLoad(pass_in, int2(id.xy), 0, 0).rgb;
    col = mix(lastFrameCol, col, 1. / float(time.frame + 1u));

    // Store to Buffer 0
    textureStore(pass_out, int2(id.xy), 0, float4(col, 1.));

    // apply exposure (how long the shutter is open)
    col *= EXPOSURE;

    // convert unbounded HDR color range to SDR color range
    col = ACESFilm(col);

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
