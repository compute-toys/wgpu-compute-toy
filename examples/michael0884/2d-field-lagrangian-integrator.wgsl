#define LENGTH 512

struct SimData
{
    data: array<array<array<vec3<f32>, LENGTH>, LENGTH>,2>,
}

#storage sim SimData
#define ITERATIONS 16
#define IN_BUF ((uint(ITERATIONS)*time.frame+dispatch.id)%2u)
#define OUT_BUF ((uint(ITERATIONS)*time.frame+dispatch.id+1u)%2u)

fn Load(i: uint2)  -> float3     { return sim.data[IN_BUF][i.x][i.y];  }
fn Store(i: uint2, data: float3) { sim.data[OUT_BUF][i.x][i.y] = data; }

fn sdSegment(p: float2, a: float2, b: float2) -> float
{
    let pa = p-a;
    let ba = b-a;
    let h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h );
}

alias Dual = vec2<f32>;

fn dC(a: float) -> Dual
{
    return Dual(a, 0.0);
}

fn dV(a: float) -> Dual
{
    return Dual(a, 1.0);
}

fn dMul(a: Dual, b: Dual) -> Dual
{
    return Dual(
        a.x * b.x,
        a.y * b.x + a.x * b.y,
    );
}

fn dChain(a: Dual, s: float, ds: float) -> Dual
{
    return Dual(
        s,
        ds * a.y,
    );
}

fn dSqrt(a: Dual) -> Dual
{
    let s = sqrt(abs(a.x));
    let ds = (0.5 / (s+0.0001));
    return dChain(a, s, ds);
}

fn dSin(a: Dual) -> Dual
{
   let s = sin(a.x);
   let ds = cos(a.x);
   return dChain(a, s, ds);
}

fn dCos(a: Dual) -> Dual
{
   let s = cos(a.x);
   let ds = -sin(a.x);
   return dChain(a, s, ds);
}

fn dSqr(a: Dual) -> Dual
{
   let s = a.x*a.x;
   let ds = 2.0*a.x;
   return dChain(a, s, ds);
}

fn Ltime(qt: Dual) -> Dual
{
    return dSqr(qt);
    //return -dSqrt(dC(1.0) - dSqr(qt));
}

fn Lspace(qx: Dual) -> Dual
{
    return -0.5*dSqr(qx);
}

fn Lpotential(q: Dual) -> Dual
{
    return custom.CosK*dCos(custom.CosA*q);
}

//a is the value we are varying
//at0 - previous timestep
//ax0,ax2 - neighbor space samples
//the part of the action that does not depend on at2
fn Action0(a: Dual, at0: Dual, ax0: Dual, ax2: Dual, ay0: Dual, ay2: Dual) -> Dual
{
    return Ltime(a - at0) +
           Lspace(a - ax0) + Lspace(ax2 - a) +
           Lspace(a - ay0) + Lspace(ay2 - a) +
           2.0*Lpotential(a);
}

//at2 - the value we want to predict
fn Action(a: Dual, at2: Dual, action0: Dual) -> Dual
{
    //all terms of the action integral that contain a
    return Ltime(at2 - a) + action0;
}

#define NEWTON_STEPS 64
#define STEP_SIZE 1.0
fn Solve(i: uint2) -> float3
{
    let a = Load(i);
    let a0 = a.x - a.y;
    let a1 = a.x;
    var a2 = a.x; //predicted value

    let f = dV(a1);
    let ft0 = dC(a0);
    let fx0 = dC(Load(i - uint2(1u, 0u)).x);
    let fx2 = dC(Load(i + uint2(1u, 0u)).x);
    let fy0 = dC(Load(i - uint2(0u, 1u)).x);
    let fy2 = dC(Load(i + uint2(0u, 1u)).x);
    let action0 = Action0(f, ft0, fx0, fx2, fy0, fy2);

    var dA = 0.0;
    var dT = STEP_SIZE;
    var j = 0;
    for(; j<NEWTON_STEPS; j++)
    {
        var A0 = Action(f, dC(a2), action0);
        let A1 = Action(f, dC(a2 - 0.001), action0);
        let A2 = Action(f, dC(a2 + 0.001), action0);
        dA = A0.y*0.002/(A2.y - A1.y);
        a2 -= dT*dA;

        if(abs(dA) < 1e-6)  { break; } //update step small enough
        if(j == 8) { dT = 1.0; } //use full step for rapid convergence
    }

    let range = 8.0;
    let avg = 0.5*(fx0.x + fx2.x);
    a2 = clamp(a2, avg - range, avg + range);
    let at = a2 - a1;

    a2 = a1+at*custom.TimeStep;
    return float3(a2,at,float(j)/32.0);
}

#dispatch_count Simulation ITERATIONS
#workgroup_count Simulation 32 32 1
@compute @workgroup_size(16,16)
fn Simulation(@builtin(global_invocation_id) id: uint3)
{
    if(time.frame == 0u)
    {
        let x = 32.0*(float2(id.xy)/float(LENGTH) - 0.5);
        let a =4.0*exp(-dot(x,x));
        let b = 4.0*exp(-dot(x - 5.0, x - 5.0));
        Store(id.xy, float3(a + b, 0.0, 0.0));
        return;
    }

    if(id.x == 0u || id.x == uint(LENGTH) - 1u || id.y == 0u || id.y == uint(LENGTH) - 1u)
    {
        Store(id.xy, float3(0.0));
        return;
    }

    let f = Solve(id.xy);
    Store(id.xy, clamp(f, float3(-35.0), float3(35.0)));
}

fn Sample(p: float2) -> float3
{
    let screen_size = float2(textureDimensions(screen));
    let id = uint2(float(LENGTH) * ((p - screen_size.xy*0.5)/float(SCREEN_HEIGHT) + 0.5));
    return Load(id);
}

@compute @workgroup_size(16, 16)
fn MainImage(@builtin(global_invocation_id) id: uint3)
{
    // Viewport resolution (in pixels)
    let screen_size = uint2(textureDimensions(screen));

    // Prevent overdraw for workgroups on the edge of the viewport
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    // Pixel coordinates (centre of pixel, origin at bottom left)
    let fragCoord = float2(float(id.x) + .5, float(screen_size.y - id.y) - .5);

    var col = float3(0.35)*pow(length(Sample(fragCoord)),1.0);

    col = pow(col, float3(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
