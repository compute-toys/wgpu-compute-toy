// This is a fork of boids-predation-4d.wgsl using a neighbourhood grid for faster nearest-neighbour lookups

#include "Dave_Hoskins/hash"

// sqrt of particle count
#define PREY 250
#define PREDATORS 20

// number of grid cells per dimension
#define GRID_RES 16

// total number of grid cells = pow(GRID_RES, 3)
#define GRID_SIZE 4096

// maximum number of particles per cell
#define GRID_CAP 1000

// world size of grid extents
#define GRID_WIDTH 50.

struct Atoms {
    pixels: array<array<array<atomic<u32>,3>, SCREEN_HEIGHT>, SCREEN_WIDTH>,
    count: array<atomic<u32>, GRID_SIZE>,
}

#storage atoms Atoms

struct Particle {
    position: float3,
    velocity: float3,
}

struct Store {
    particles: array<array<Particle, GRID_CAP>, GRID_SIZE>,
    count: array<u32, GRID_SIZE>,
}

#storage store Store



fn hue(v: float) -> float4 {
    return .6 + .6 * cos(6.3 * v + float4(0.,23.,21.,0.));
}

fn clamp_length(v: float3, r: float) -> float3 {
    if (length(v) > r) {
        return r * normalize(v);
    } else {
        return v;
    }
}

fn normz(v: float3) -> float3 {
    if (length(v) == 0.) {
        return float3(0.);
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

fn camera() -> Camera {
    let screen_size = int2(textureDimensions(screen));

    var ang = float2(mouse.pos) * float2(-radians(360.), radians(180.)) / float2(screen_size);
    if (ang.y == 0.) {
        ang.y = radians(120.);
    }
    ang += 1e-4;
    ang.x += custom.Rotation * time.elapsed;

    var camera = Camera();
    camera.fov = .8;
    camera.cam = GetCameraMatrix(ang);
    camera.pos = camera.cam * -float3(100. * custom.Zoom, 0., 0.);
    camera.size = float2(textureDimensions(screen));
    return camera;
}

fn RasterizePoint(p: float3, color: float3, r: int) {
    let screen_size = int2(textureDimensions(screen));
    let cam = camera();

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
            //let c = 255. * color / pos.z;
            let c = 255. * hue(pos.z / GRID_WIDTH).rgb;
            atomicAdd(&atoms.pixels[int(pos.x) + i][int(pos.y) + j][0], uint(c.x));
            atomicAdd(&atoms.pixels[int(pos.x) + i][int(pos.y) + j][1], uint(c.y));
            atomicAdd(&atoms.pixels[int(pos.x) + i][int(pos.y) + j][2], uint(c.z));
        }
    }
}

fn Sample(pos: int2) -> float3 {
    let screen_size = int2(textureDimensions(screen));
    let idx = pos.x + screen_size.x * pos.y;
    let x = float(atomicLoad(&atoms.pixels[pos.x][pos.y][0]));
    let y = float(atomicLoad(&atoms.pixels[pos.x][pos.y][1]));
    let z = float(atomicLoad(&atoms.pixels[pos.x][pos.y][2]));
    return float3(x,y,z) / 255.;
}

fn LoadParticle(id: int2) -> Particle {
    var p = Particle();
    p.position = passLoad(0, id, 0).xyz;
    p.velocity = passLoad(1, id, 0).xyz;
    return p;
}

fn SaveParticle(id: int2, p: Particle) {
    passStore(0, id, float4(p.position, 0.));
    passStore(1, id, float4(p.velocity, 0.));
}

fn grid_cell(pos: float3) -> int3 {
    return clamp(int3((.5 + pos / GRID_WIDTH) * float(GRID_RES)), int3(0), int3(GRID_RES - 1));
}

fn grid_cell_id(cell: int3) -> int {
    return cell.x + cell.y * GRID_RES + cell.z * GRID_RES * GRID_RES;
}

fn grid_id_to_cell(i: int) -> int3 {
    return int3(i, i / GRID_RES, i / GRID_RES / GRID_RES) % GRID_RES;
}

fn grid_cell_dist_min(cell1: int3, cell2: int3) -> float {
    let cell_width = GRID_WIDTH / float(GRID_RES);
    let cdist = distance(float3(cell1), float3(cell2)) * cell_width;
    return max(0., cdist - sqrt(3.) * cell_width);
}

fn grid_increment(cell_id: int) -> int {
    return int(atomicAdd(&atoms.count[cell_id], 1u));
}

fn grid_clear(cell_id: int) {
    atomicStore(&atoms.count[cell_id], 0u);
}

fn grid_insert(p: Particle) {
    let cell = grid_cell(p.position);
    let cell_id = grid_cell_id(cell);

    let idx = grid_increment(cell_id);
    if (idx >= GRID_CAP) { return; }

    store.particles[cell_id][idx] = p;
}

fn grid_neighbours() -> array<int3, 27> {
    var r: array<int3, 27>;
    var n = 0;
    for (var i = -1; i <= 1; i += 1) {
        for (var j = -1; j <= 1; j += 1) {
            for (var k = -1; k <= 1; k += 1) {
                r[n] = int3(i,j,k);
                n += 1;
            }
        }
    }
    return r;
}

@compute @workgroup_size(16, 16)
fn SimulatePrey(@builtin(global_invocation_id) id: uint3) {
    var p = LoadParticle(int2(id.xy));

    if(int(id.x) > PREY || int(id.y) > PREY) {
        return;
    }

    if(time.frame == 0u) {
        p.position = (hash42(float2(id.xy)).xyz - .5) * GRID_WIDTH;
    }

    var separation = float3();
    var mean = Particle();
    var count = 0;

    let offsets = grid_neighbours();

    let cell = grid_cell(p.position);

    // other prey
    for(var o = 0; o < 27; o += 1) {
        let offset = offsets[o];
        let ncell = cell + offset;
        if (any(ncell < int3(0)) || any(ncell >= int3(GRID_RES))) {
            continue;
        }
        let i = grid_cell_id(ncell);
        let gcount = int(store.count[i]);
        for(var j = 0; j < gcount; j += 1) {
            let q = store.particles[i][j];
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
            if (length(d) < custom.DistPrey * 30.) {
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
        p.position = (hash42(float2(id.xy)).xyz - .5) * GRID_WIDTH;
    }

    var separation = float3();

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

    let cell = grid_cell(p.position);

    // prey
    for(var i = 0; i < GRID_SIZE; i += 1) {
        let ncell = grid_id_to_cell(i);
        var gcount = int(store.count[i]);
        for(var j = 0; j < 10; j += 1) {
            if (j >= gcount) {
                break;
            }
            let q = store.particles[i][j];
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
    atomicStore(&atoms.pixels[id.x][id.y][0], 0u);
    atomicStore(&atoms.pixels[id.x][id.y][1], 0u);
    atomicStore(&atoms.pixels[id.x][id.y][2], 0u);

    if (id.x == 0u && id.y == 0u && id.z == 0u) {
        for (var i = 0; i < GRID_SIZE; i += 1) {
            grid_clear(i);
        }
    }
}

@compute @workgroup_size(16, 16)
fn BuildGrid(@builtin(global_invocation_id) id: uint3) {
    var p = LoadParticle(int2(id.xy));

    if(int(id.x) > PREY || int(id.y) > PREY) {
        return;
    }

    grid_insert(p);
}

@compute @workgroup_size(256)
#workgroup_count FreezeGrid 16 1 1
fn FreezeGrid(@builtin(global_invocation_id) id: uint3) {
    if(int(id.x) >= GRID_SIZE || int(id.y) > 0) {
        return;
    }
    let cell_id = int(id.x);
    store.count[cell_id] = uint(atomicLoad(&atoms.count[cell_id]));
}

@compute @workgroup_size(16, 16)
#workgroup_count RasterizePrey 64 256 1
fn RasterizePrey(@builtin(global_invocation_id) id: uint3) {
    if (int(id.y) < GRID_SIZE && id.x < store.count[id.y]) {
        let p = store.particles[id.y][id.x];
        RasterizePoint(p.position.xyz, float3(1.), 1);
    }
}

@compute @workgroup_size(16, 16)
fn RasterizePredators(@builtin(global_invocation_id) id: uint3) {
    var r = 0;
    if (int(id.x) <= PREY && int(id.y) <= PREY) {
        return;
    } else if (int(id.x) <= PREY + PREDATORS && int(id.y) <= PREDATORS) {
        r = 3;
    } else {
        return;
    }

    let p = LoadParticle(int2(id.xy));
    RasterizePoint(p.position.xyz, float3(1.), r);
}

@compute @workgroup_size(16, 16)
fn Accumulate(@builtin(global_invocation_id) id: uint3) {
    let screen_size = uint2(textureDimensions(screen));
    var col = .1 * Sample(int2(id.xy));
    if (mouse.click == 0) {
        col += passLoad(2, int2(id.xy), 0).rgb * custom.Accumulation;
    }
    passStore(2, int2(id.xy), float4(col, 1.));

    if (int(id.x) < GRID_SIZE && screen_size.y - id.y < store.count[id.x]) {
        let p = store.particles[id.x][id.y];
        col += .1;
    }

    textureStore(screen, int2(id.xy), float4(col, 1.));
}
