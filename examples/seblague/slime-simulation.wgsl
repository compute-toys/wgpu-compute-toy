// This is a port of https://playground.babylonjs.com/?webgpu#GXJ3FZ#48 which itself is a port of https://github.com/SebLague/Slime-Simulation (GPLv3)

#include "Dave_Hoskins/hash"

#define NUM_AGENTS 256000

struct Agent {
	position : vec2<f32>,
	angle : f32,
	species : i32,
};

struct Store {
    agents: array<Agent, NUM_AGENTS>,
    trail: array<array<vec4<f32>, SCREEN_HEIGHT>, SCREEN_WIDTH>,
}

#storage store Store

struct Settings {
	turnSpeed : f32,
	sensorAngleDegrees : f32,
	sensorOffsetDst : f32,
};

fn getSettings(i: int) -> Settings {
    if (i == 0) {
        return Settings(
            100. * custom.TurnSpeed1,
            180. * custom.SensorAngle1,
            50. * custom.SensorOffset1,
        );
    } else if (i == 1) {
        return Settings(
            100. * custom.TurnSpeed2,
            180. * custom.SensorAngle2,
            50. * custom.SensorOffset2,
        );
    }
    return Settings();
}

const speciesMask = array<float4, 4>(
    float4(1.,0.,0.,0.),
    float4(0.,1.,0.,0.),
    float4(0.,0.,1.,0.),
    float4(0.,0.,0.,1.),
);

@compute @workgroup_size(256)
#workgroup_count Init 1000 1 1
fn Init(@builtin(global_invocation_id) id : vec3<u32>)
{
    if(time.frame > 0) { return; }

    let h = hash41(f32(id.x));
    store.agents[id.x] = Agent(
        float2(textureDimensions(screen)) * h.xy,
        radians(360.) * h.z,
        int(2. * h.w),
    );
}

// Hash function www.cs.ubc.ca/~rbridson/docs/schechter-sca08-turbulence.pdf
fn hash(state0 : u32) -> u32
{
    var state = state0;
    state = state ^ 2747636419u;
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    state = state ^ (state >> 16u);
    state = state * 2654435769u;
    return state;
}

fn scaleToRange01(state : u32) -> f32
{
    return f32(state) / 4294967295.0;
}

fn sense(agent : Agent, settings : Settings, sensorAngleOffset : f32) -> f32
{
    let screen_size = int2(textureDimensions(screen));

	let sensorAngle = agent.angle + sensorAngleOffset;
	let sensorDir = vec2<f32>(cos(sensorAngle), sin(sensorAngle));

	let sensorPos = agent.position + sensorDir * settings.sensorOffsetDst;
	let sensorCentreX = i32(sensorPos.x);
	let sensorCentreY = i32(sensorPos.y);

	var sum = 0.;

	let senseWeight = speciesMask[agent.species] * (1. + custom.SpeciesAvoidance) - custom.SpeciesAvoidance;

    let sensorSize = 1;
	for (var offsetX = -sensorSize; offsetX <= sensorSize; offsetX = offsetX + 1) {
		for (var offsetY = -sensorSize; offsetY <= sensorSize; offsetY = offsetY + 1) {
			let sampleX = min(screen_size.x - 1, max(0, sensorCentreX + offsetX));
			let sampleY = min(screen_size.y - 1, max(0, sensorCentreY + offsetY));
			sum = sum + dot(vec4<f32>(senseWeight), store.trail[sampleX][sampleY]);
		}
	}

	return sum;
}

@compute @workgroup_size(256)
#workgroup_count Simulation 1000 1 1
fn Simulation(@builtin(global_invocation_id) id : vec3<u32>)
{
    let screen_size = textureDimensions(screen);

	let agent = store.agents[id.x];
	let settings = getSettings(agent.species);
	let pos = agent.position;

	var random = hash(u32(pos.y * f32(screen_size.x) + pos.x) + hash(id.x + u32(time.elapsed * 100000.)));

	// Steer based on sensory data
	let sensorAngleRad = settings.sensorAngleDegrees * (3.1415 / 180.);
	let weightForward = sense(agent, settings, 0.);
	let weightLeft = sense(agent, settings, sensorAngleRad);
	let weightRight = sense(agent, settings, -sensorAngleRad);


	let randomSteerStrength = scaleToRange01(random);
	let turnSpeed = settings.turnSpeed * 2. * 3.1415;

	// Continue in same direction
	if (weightForward > weightLeft && weightForward > weightRight) {
		store.agents[id.x].angle += 0.;
	}
	else if (weightForward < weightLeft && weightForward < weightRight) {
		store.agents[id.x].angle += (randomSteerStrength - 0.5) * 2. * turnSpeed * time.delta;
	}
	// Turn right
	else if (weightRight > weightLeft) {
		store.agents[id.x].angle -= randomSteerStrength * turnSpeed * time.delta;
	}
	// Turn left
	else if (weightLeft > weightRight) {
		store.agents[id.x].angle += randomSteerStrength * turnSpeed * time.delta;
	}

	// Update position
	let direction = vec2<f32>(cos(agent.angle), sin(agent.angle));
	var newPos = agent.position + direction * time.delta * 200. * custom.MoveSpeed;

	// Clamp position to map boundaries, and pick new random move dir if hit boundary
	if (newPos.x < 0. || newPos.x >= f32(screen_size.x) || newPos.y < 0. || newPos.y >= f32(screen_size.y)) {
		random = hash(random);
		let randomAngle = scaleToRange01(random) * 2. * 3.1415;

		newPos.x = min(f32(screen_size.x - 1), max(0., newPos.x));
		newPos.y = min(f32(screen_size.y - 1), max(0., newPos.y));
		store.agents[id.x].angle = randomAngle;
	} else {
		let oldTrail = store.trail[i32(newPos.x)][i32(newPos.y)];
        let newVal = min(vec4<f32>(1.), oldTrail + speciesMask[agent.species] * 20. * custom.TrailWeight * time.delta);
		store.trail[i32(newPos.x)][i32(newPos.y)] = newVal;
	}

	store.agents[id.x].position = newPos;
}

@compute @workgroup_size(16, 16)
fn Diffuse(@builtin(global_invocation_id) id : vec3<u32>)
{
    let screen_size = int2(textureDimensions(screen));
	if (id.x >= u32(screen_size.x) || id.y >= u32(screen_size.y)) {
		return;
	}

	var sum = vec4<f32>(0.);
	let originalCol = store.trail[id.x][id.y];
	// 3x3 blur
	for (var offsetX = -1; offsetX <= 1; offsetX = offsetX + 1) {
		for (var offsetY = -1; offsetY <= 1; offsetY = offsetY + 1) {
			let sampleX = min(screen_size.x - 1, max(0, i32(id.x) + offsetX));
			let sampleY = min(screen_size.y - 1, max(0, i32(id.y) + offsetY));
			sum = sum + store.trail[sampleX][sampleY];
		}
	}

	var blurredCol = sum / vec4<f32>(9., 9., 9., 9.);
	let diffuseWeight = clamp(5. * custom.DiffuseRate * time.delta, 0., 1.);

	blurredCol = originalCol * (1. - diffuseWeight) + blurredCol * diffuseWeight;

    let p = 2. * custom.DecayRate * time.delta;
    let pix = max(vec4<f32>(0., 0., 0., 0.), blurredCol - vec4<f32>(p));

    store.trail[id.x][id.y] = pix;
}

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: uint3) {
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    var col = store.trail[id.x][id.y].rgb;

    // Convert from gamma-encoded to linear colour space
    col = pow(col, float3(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
