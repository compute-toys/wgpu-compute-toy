// A four-dimensional boids predator/prey model. The particles are projected to three dimensions (click and drag the mouse to rotate) with the hue indicating the position in the fourth dimension.

#storage atomic_storage array<atomic<i32>>

// sqrt of particle count
const PREY = 100;
const PREDATORS = 20;

fn hue(v: float) -> float4 {
    return .6 + .6 * cos(6.3 * v + float4(0.,23.,21.,0.));
}

fn hash42(p: float2) -> float4 {
    var p4 = fract(float4(p.xyxy) * float4(.1031, .1030, .0973, .1099));
    p4 += dot(p4, p4.wzxy+33.33);
    return fract((p4.xxyz+p4.yzzw)*p4.zywx);
}

fn clamp_length(v: float4, r: float) -> float4 {
    if (length(v) > r) {
        return r * normalize(v);
    } else {
        return v;
    }
}

fn normz(v: float4) -> float4 {
    if (length(v) == 0.) {
        return float4(0.);
    } else {
        return normalize(v);
    }
}

// particle rendering code based on "3D atomic rasterizer" by michael0884
// https://compute.toys/view/21

struct Camera {
    pos: float3,
    cam: float3x3,
    fov: float,
    size: float2,
}

fn GetCameraMatrix(ang: float2) -> float3x3
{
    let x_dir = float3(cos(ang.x)*sin(ang.y), cos(ang.y), sin(ang.x)*sin(ang.y));
    let y_dir = normalize(cross(x_dir, float3(0.,1.,0.)));
    let z_dir = normalize(cross(y_dir, x_dir));
    return float3x3(-x_dir, y_dir, z_dir);
}

fn RasterizePoint(cam: Camera, p: float3, color: float3, r: int) {
    let screen_size = int2(textureDimensions(screen));

    // project to clip space
    let dir = normalize(p - cam.pos);
    let screen = dir * cam.cam;
    let pos = float3(screen.yz * cam.size.y / (cam.fov * screen.x) + .5 * cam.size, screen.x * distance(cam.pos, p));
    if (pos.x < 0. || pos.x > cam.size.x || pos.y < 0. || pos.y > cam.size.y || pos.z < 0.) {
        return;
    }

    for (var i = -r; i <= r; i += 1) {
        for (var j = -r; j <= r; j += 1) {
            let idx = int(pos.x) + i + screen_size.x * (int(pos.y) + j);
            let c = 255. * color / pos.z;
            atomicAdd(&atomic_storage[idx*4+0], int(c.x));
            atomicAdd(&atomic_storage[idx*4+1], int(c.y));
            atomicAdd(&atomic_storage[idx*4+2], int(c.z));
        }
    }
}

fn Sample(pos: int2) -> float3 {
    let screen_size = int2(textureDimensions(screen));
    let idx = pos.x + screen_size.x * pos.y;
    let x = float(atomicLoad(&atomic_storage[idx*4+0]));
    let y = float(atomicLoad(&atomic_storage[idx*4+1]));
    let z = float(atomicLoad(&atomic_storage[idx*4+2]));
    return float3(x,y,z) / 255.;
}

struct Particle {
    position: float4,
    velocity: float4,
}

fn LoadParticle(id: int2) -> Particle {
    var p = Particle();
    p.position = passLoad(0, id, 0);
    p.velocity = passLoad(1, id, 0);
    return p;
}

fn SaveParticle(id: int2, p: Particle) {
    passStore(0, id, p.position);
    passStore(1, id, p.velocity);
}

@compute @workgroup_size(16, 16)
fn SimulatePrey(@builtin(global_invocation_id) id: uint3) {
    var p = LoadParticle(int2(id.xy));

    if(int(id.x) > PREY || int(id.y) > PREY) {
        return;
    }

    if(time.frame == 0u) {
        p.position = hash42(float2(id.xy));
    }

    var separation = float4();
    var mean = Particle();
    var count = 0;

    // other prey
    for(var i = 0; i < PREY; i += 1) {
        for(var j = 0; j < PREY; j += 1) {
            let q = LoadParticle(int2(i,j));
            let d = p.position - q.position;
            if (length(d) < 1e-3) {
                continue;
            }
            if (length(d) < custom.DistSeparation * 30.) {
                separation += d / pow(length(d), 2.);
            }
            if (length(d) < custom.DistCohesion * 30.) {
                mean.position += q.position;
                mean.velocity += q.velocity;
                count += 1;
            }
        }
    }

    // predators
    for(var i = PREY; i < PREY + PREDATORS; i += 1) {
        for(var j = 0; j < PREDATORS; j += 1) {
            let q = LoadParticle(int2(i,j));
            let d = p.position - q.position;
            if (length(d) < 1e-3) {
                continue;
            }
            if (length(d) < custom.DistPrey * 30.)
            {
                separation += d / pow(length(d), 1.);
            }
        }
    }

    if (count > 0) {
        mean.position /= float(count);
        mean.velocity /= float(count);
    }

    var cohesion = mean.position - p.position;
    var alignment = mean.velocity;

    let v = p.velocity;
    p.velocity += clamp_length(normz(separation) - v, custom.MaxForce * .1) * 1.5;
    p.velocity += clamp_length(normz(cohesion)   - v, custom.MaxForce * .1);
    p.velocity += clamp_length(normz(alignment)  - v, custom.MaxForce * .1);

    if (length(p.position) > 20.) {
        p.velocity -= pow(10., -10. * custom.Homing) * normalize(p.position);
    }

    p.velocity = clamp_length(p.velocity, custom.MaxSpeed);

    p.position += custom.Timescale * p.velocity;

    SaveParticle(int2(id.xy), p);
}

@compute @workgroup_size(16, 16)
fn SimulatePredators(@builtin(global_invocation_id) id: uint3) {
    var p = LoadParticle(int2(id.xy));

    if(int(id.x) <= PREY || int(id.x) > PREY + PREDATORS || int(id.y) > PREDATORS) {
        return;
    }

    if(time.frame == 0u) {
        p.position = hash42(float2(id.xy));
    }

    var separation = float4();

    // other predators
    for(var i = PREY; i < PREY + PREDATORS; i += 1) {
        for(var j = 0; j < PREDATORS; j += 1) {
            let q = LoadParticle(int2(i,j));
            let d = p.position - q.position;
            if (length(d) < 1e-3) {
                continue;
            }
            if (length(d) < custom.DistSeparationPred * 30.) {
                separation += d / pow(length(d), 1.);
            }
        }
    }

    // prey
    for(var i = 0; i < PREY; i += 1) {
        for(var j = 0; j < PREY; j += 1) {
            let q = LoadParticle(int2(i,j));
            let d = p.position - q.position;
            if (length(d) < 1e-3) {
                continue;
            }
            if (length(d) < custom.DistPredation * 30.) {
                separation -= d / pow(length(d), 1.);
            }
        }
    }

    p.velocity = clamp_length(p.velocity + clamp_length(separation, custom.PredForce * .1), custom.PredSpeed);

    p.position += custom.Timescale * p.velocity;

    SaveParticle(int2(id.xy), p);
}

@compute @workgroup_size(16, 16)
fn Clear(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    let idx = int(id.x + screen_size.x * id.y);
    atomicStore(&atomic_storage[idx*4+0], 0);
    atomicStore(&atomic_storage[idx*4+1], 0);
    atomicStore(&atomic_storage[idx*4+2], 0);
    atomicStore(&atomic_storage[idx*4+3], 0);
}

@compute @workgroup_size(16, 16)
fn Rasterize(@builtin(global_invocation_id) id: uint3) {
    let screen_size = int2(textureDimensions(screen));

    var ang = float2(mouse.pos) * float2(-radians(360.), radians(180.)) / float2(screen_size) + 1e-4;
    ang.x += custom.Rotation * time.elapsed;

    var camera = Camera();
    camera.fov = .8;
    camera.cam = GetCameraMatrix(ang);
    camera.pos = camera.cam * -float3(100. * custom.Zoom, 0., 0.);
    camera.size = float2(textureDimensions(screen));

    var r = 0;
    if (int(id.x) <= PREY && int(id.y) <= PREY) {
        r = 1;
    } else if (int(id.x) <= PREY + PREDATORS && int(id.y) <= PREDATORS) {
        r = 3;
    } else {
        return;
    }

    let p = LoadParticle(int2(id.xy));
    RasterizePoint(camera, p.position.xyz, hue(.5 + .5 * p.position.w / 20.).rgb, r);
}

@compute @workgroup_size(16, 16)
fn Accumulate(@builtin(global_invocation_id) id: uint3) {
    var col = 20. * Sample(int2(id.xy));
    if (mouse.click == 0) {
        col += passLoad(2, int2(id.xy), 0).rgb * custom.Accumulation;
    }
    passStore(2, int2(id.xy), float4(col, 1.));
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
