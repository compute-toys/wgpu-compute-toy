// 1+1 relativistic Sine Gordon string lagrangian

#define LENGTH 512

struct SimData
{
    data: array<array<vec3<f32>, LENGTH>,2>,
}

#storage sim SimData

#define IN_BUF (time.frame%2u)
#define OUT_BUF ((time.frame+1u)%2u)

fn Load(i: uint)  -> float3     { return sim.data[IN_BUF][i];  }
fn Store(i: uint, data: float3) { sim.data[OUT_BUF][i] = data; }

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
    //return 0.5*dSqr(qt);
    return -dSqrt(dC(1.0) - dSqr(qt));
}

fn Lspace(qx: Dual) -> Dual
{
    return -0.5*dSqr(qx);
}

fn Lpotential(q: Dual) -> Dual
{
    return  custom.CosK*dCos(custom.CosA*q);
}

fn Dt(a0: Dual, a1: Dual) -> Dual
{
    return (a1 - a0)/custom.SpeedLimit;
}

fn Dx(a0: Dual, a1: Dual) -> Dual
{
    return a1 - a0;
}

//a is the value we are varying
//at0 - previous timestep
//at2 - the value we want to predict
//ax0,ax2 - neighbor space samples
fn Action(a: Dual, at0: Dual, at2: Dual, ax0: Dual, ax2: Dual) -> Dual
{
    //all terms of the action integral that contain a1
    return Ltime(a - at0) + Ltime(at2 - a) +
           Lspace(a - ax0) + Lspace(ax2 - a) +
           2.0*Lpotential(a);
}

#define NEWTON_STEPS 64
#define STEP_SIZE 0.25
fn Solve(i: uint) -> float3
{
    let a = Load(i);
    let a0 = a.x - a.y;
    let a1 = a.x;
    var a2 = a.x; //predicted value

    let f = dV(a1);
    let ft0 = dC(a0);
    let fx0 = dC(Load(i - 1u).x);
    let fx2 = dC(Load(i + 1u).x);
    var dA = 0.0;
    var dT = STEP_SIZE;
    var a2prev = a2;
    var A0 = Action(f, ft0, dC(a2), fx0, fx2);
    var j = 0;
    var K = 0.0;
    for(; j<NEWTON_STEPS; j++)
    {
        let A1 = Action(f, ft0, dC(a2 - 0.001), fx0, fx2);
        let A2 = Action(f, ft0, dC(a2 + 0.001), fx0, fx2);
        dA = A0.y*0.002/(A2.y - A1.y);

        a2prev = a2;
        a2 = a2prev - dT*dA;

        if(abs(dA) < 1e-6)  { break; } //update step small enough

        let A0prev = A0;
        A0 = Action(f, ft0, dC(a2), fx0, fx2);

        if(j == 3) { dT = 1.0; } //use full step for rapid convergence
    }

    let range = 2.0;
    let avg = 0.5*(fx0.x + fx2.x);
    a2 = clamp(a2, avg - range, avg + range);
    let at = a2 - a1;
   // let at = a.y + (fx0.x + fx2.x - 2.0*a1)*custom.TimeStep;
    a2 = a1+at*custom.TimeStep;
    return float3(a2,at,float(j)/32.0);
}

#workgroup_count Simulation 2 1 1
@compute @workgroup_size(256)
fn Simulation(@builtin(global_invocation_id) id: uint3)
{
    if(time.frame == 0u)
    {
        let x = 32.0*(float(id.x)/float(LENGTH) - 0.5);
        Store(id.x, float3(12.0*exp(-x*x), 0.0, 0.0));
        return;
    }

    if(id.x != 0u && id.x != uint(LENGTH) - 1u)
    {
        let f = Solve(id.x);
        Store(id.x, clamp(f, float3(-35.0), float3(35.0)));
    }
}

fn GetPoint(i: uint) -> float2
{
    return float2(float(SCREEN_WIDTH) * (float(i) + 0.5)/ float(LENGTH), Load(i).x * float(SCREEN_HEIGHT)/30.0 + float(SCREEN_HEIGHT) * 0.5);
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
    var am = 0.0;
    for(var i = 0u; i < (uint(LENGTH) - 1u); i++)
    {
        let sdSeg = sdSegment(fragCoord, GetPoint(i), GetPoint(i+1u));
        if(sdSeg < sdf)
        {
            sdf = sdSeg;
            am = Load(i).z;
        }
    }

    var col = float3(1.0, 1.0, 1.0)*smoothstep(2.0, 1.0, sdf) +
              float3(1.0, 0.0, 0.0)*abs(am)*smoothstep(12.0, 10.0, sdf);

    col = pow(col, float3(2.2));

    // Output to screen (linear colour space)
    textureStore(screen, int2(id.xy), float4(col, 1.));
}
