// Using the lagrangian to solve for the discrete equations of motion.

#define LENGTH 256

#storage sim_data array<array<float, LENGTH>, 2>

#define IN_BUF (time.frame%2u)
#define OUT_BUF ((time.frame+1u)%2u)

fn Load(i: uint)  -> float     { return sim_data[IN_BUF][i];  }
fn Store(i: uint, data: float) { sim_data[OUT_BUF][i] = data; }

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

//relativistic harmonic oscillator
fn Lagrangian(q: Dual, qt: Dual) -> Dual
{
    return -dSqrt(dC(1.0) - dSqr(qt)) - 0.1*dSqr(q);
}

fn DLagrangian(a0: Dual, a1: Dual) -> Dual
{
    let Dt = 1.0*(a1 - a0);
    return Lagrangian(a0, Dt) + Lagrangian(a1, Dt);
}

fn Action(a0: Dual, a1: Dual, a2: Dual) -> Dual
{
    return DLagrangian(a0, a1) + DLagrangian(a1, a2);
}

#define NEWTON_STEPS 16
#define STEP_SIZE 3e-1
fn Solve(i: uint) -> float
{
    let a0 = Load(i - 2u);
    let a1 = Load(i - 1u);
    var a2 = 2.0*a1 - a0;
    let f0 = dC(a0);
    let f1 = dV(a1);

    for(var i = 0; i<NEWTON_STEPS; i++)
    {
        let A0 = Action(f0,f1,dC(a2));
        let A1 = Action(f0,f1,dC(a2 - 0.005));
        let A2 = Action(f0,f1,dC(a2 + 0.005));
        let dA = (A2.y - A1.y)/0.01;
        a2 = a2 - STEP_SIZE*A0.y/dA;
    }

    return a2;
}

#workgroup_count Simulation 1 1 1
@compute @workgroup_size(LENGTH)
fn Simulation(@builtin(global_invocation_id) id: uint3)
{
    if(time.frame == 0u)
    {
        if(id.x < 4u)
        {
            Store(id.x, 12.0);
        }
        return;
    }

    var f = Load(id.x);

    if(id.x >= 2u && id.x == time.frame)
    {
       f = Solve(id.x);
    }

    Store(id.x, clamp(f, -35.0, 35.0));
}

fn GetPoint(i: uint) -> float2
{
    return float2(float(SCREEN_WIDTH) * float(i) / float(LENGTH), 10.0*Load(i) + float(SCREEN_HEIGHT) * 0.5);
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

    var sdf = 1e10;
    for(var i = 0u; i < (uint(LENGTH) - 1u); i++)
    {
        sdf = min(sdf, sdSegment(fragCoord, GetPoint(i), GetPoint(i+1u)));
    }

    var col = float3(1.0)*smoothstep(1.0, 2.0, sdf);

    col = pow(col, float3(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
